"""
RT-2 Simulation Starter (single-file)

What this does:
- Creates a PyBullet scene (plane + KUKA iiwa + 3 colored cubes)
- Captures camera image and segmentation mask
- Accepts free-form natural language commands from the user (console input)
- Uses Sentence-Transformer embeddings to semantically match parts of the command to the visible objects
- Computes a simple pick-and-place using inverse kinematics and a fixed grasp constraint

How to run (Windows + VSCode):
1. Create & activate a python venv
2. pip install pybullet sentence-transformers numpy pillow opencv-python
3. Run: python rt2_sim_starter.py

Notes:
- This script avoids hardcoding command strings. Instead it generates candidate textual descriptions for each visible object
  and uses semantic similarity to decide which object the user refers to for `pick` and `place` roles.
- It's a starter: improve perception (CLIP), parsing (LLM), or control (more robust grasps) as next steps.

"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import io

# --------------------------- Configuration ---------------------------
SIM_WIDTH = 320
SIM_HEIGHT = 240
MODEL_NAME = 'all-MiniLM-L6-v2'  # small and fast sentence-transformer

# End-effector index for KUKA iiwa in this URDF (commonly 6). If you use a different URDF, inspect joint info.
EE_LINK_INDEX = 6

# --------------------------- Helper utilities ---------------------------

def connect_pybullet(gui=True):
    if gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)


def create_scene():
    plane = p.loadURDF('plane.urdf')
    # load KUKA iiwa
    robot = p.loadURDF('kuka_iiwa/model.urdf', useFixedBase=True)

    # Create colored cubes (we keep their "color name" as metadata but the system will use semantic matching)
    cube_colors = [([1,0,0,1], 'red'), ([0,1,0,1], 'green'), ([0,0,1,1], 'blue')]
    base_x = 0.55
    cube_ids = {}
    positions = [[base_x, -0.15, 0.03], [base_x, 0.0, 0.03], [base_x, 0.15, 0.03]]
    for (col, name), pos in zip(cube_colors, positions):
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03,0.03,0.03], rgbaColor=col)
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03,0.03,0.03])
        uid = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=collision,
                                baseVisualShapeIndex=visual, basePosition=pos)
        cube_ids[uid] = {'name': name, 'color': col}

    return robot, cube_ids


def get_camera_image_and_seg():
    # Camera placed to see the table area and objects
    cam_target = [0.6, 0, 0.05]
    distance = 0.9
    yaw = 90
    pitch = -20
    roll = 0
    up_axis_index = 2
    view = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=cam_target,
                                              distance=distance, yaw=yaw, pitch=pitch, roll=roll,
                                              upAxisIndex=up_axis_index)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=SIM_WIDTH/SIM_HEIGHT, nearVal=0.01, farVal=3.1)
    w, h, px, depth, seg = p.getCameraImage(SIM_WIDTH, SIM_HEIGHT, viewMatrix=view, projectionMatrix=proj,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
    # px is a flat list, convert to numpy (h,w,4)
    img = np.reshape(px, (h, w, 4))[:, :, :3]
    seg = np.array(seg).reshape((h, w))
    return img, seg


def seg_to_object_crops(img, seg, object_uids):
    """Return a dict uid -> cropped RGB numpy array (tight bounding box from segmentation)."""
    crops = {}
    h, w, _ = img.shape
    for uid in object_uids:
        mask = (seg == uid)
        if mask.sum() == 0:
            continue
        ys, xs = np.where(mask)
        y0, y1 = max(0, ys.min()-2), min(h-1, ys.max()+2)
        x0, x1 = max(0, xs.min()-2), min(w-1, xs.max()+2)
        crop = img[y0:y1+1, x0:x1+1, :]
        # If crop is empty fallback to tiny center crop
        if crop.size == 0:
            cy, cx = h//2, w//2
            crop = img[cy-16:cy+16, cx-16:cx+16, :]
        crops[uid] = crop
    return crops


# --------------------------- Semantic matching (Language -> Objects) ---------------------------

class SemanticMatcher:
    def __init__(self, model_name=MODEL_NAME):
        print('Loading sentence-transformer model...')
        self.model = SentenceTransformer(model_name)

    def pick_and_place_from_text(self, text, objects_meta):
        """
        Given full command text and objects_meta (dict uid->{'name',...}), decide which object to pick and where to place.

        Strategy:
        - Try to find explicit "pick" and "place/on/put" phrases via regex. Extract short text fragments for pick/place.
        - If not found, fallback to semantics: find top-2 objects whose candidate descriptions have highest cosine similarity to the full text,
          then choose first as pick and second as place (if obvious placement phrase like 'on the <obj>' exists we'll prefer that mapping).
        """
        # Normalize text
        txt = text.lower()
        # 1) Try regex extraction
        pick_phrase = None
        place_phrase = None

        # common verbs for pick and place
        pick_match = re.search(r"(?:pick|grab|take|lift|pick up)\s+(?:the\s+)?(.+?)(?:\s+(?:and|then|,)|$)", txt)
        place_match = re.search(r"(?:place|put|set)\s+(?:it\s+)?(?:on|onto|in|into|at)?\s*(?:the\s+)?(.+?)(?:\s+(?:and|then|,)|$)", txt)

        if pick_match:
            pick_phrase = pick_match.group(1).strip()
        if place_match:
            place_phrase = place_match.group(1).strip()

        # Build candidate descriptions for each object
        uid_list = list(objects_meta.keys())
        candidates = []
        uid_to_texts = {}
        for uid, meta in objects_meta.items():
            color_name = meta.get('name', 'object')
            # create several candidate textual descriptions for flexibility
            texts = [f"{color_name} cube", f"{color_name} block", f"a {color_name} object", f"{color_name} colored cube"]
            uid_to_texts[uid] = texts
            candidates.extend(texts)

        # Prepare embeddings
        # We'll embed the user fragments and all candidate descriptions
        lookup_texts = []
        # maintain mapping from candidate index -> uid
        cand_index_to_uid = []
        for uid, texts in uid_to_texts.items():
            for t in texts:
                lookup_texts.append(t)
                cand_index_to_uid.append(uid)

        # Texts to embed
        texts_to_embed = [txt]
        if pick_phrase:
            texts_to_embed.append(pick_phrase)
        if place_phrase and place_phrase != pick_phrase:
            texts_to_embed.append(place_phrase)
        texts_to_embed.extend(lookup_texts)

        emb = self.model.encode(texts_to_embed, convert_to_tensor=True)
        user_emb = emb[0]
        idx = 1
        pick_emb = None
        place_emb = None
        if pick_phrase:
            pick_emb = emb[idx]; idx += 1
        if place_phrase:
            place_emb = emb[idx]; idx += 1

        cand_embs = emb[idx:]

        # Compute similarity between fragment and candidates
        def best_uid_for(phrase_emb):
            sims = util.cos_sim(phrase_emb, cand_embs)[0]
            best_idx = int(sims.argmax())
            return cand_index_to_uid[best_idx], float(sims[best_idx])

        pick_uid = None
        place_uid = None
        debug_scores = {}

        if pick_emb is not None:
            pick_uid, pick_score = best_uid_for(pick_emb)
            debug_scores['pick_score'] = pick_score
        if place_emb is not None:
            place_uid, place_score = best_uid_for(place_emb)
            debug_scores['place_score'] = place_score

        # Fallback semantic method if regex didn't extract both
        if pick_uid is None or place_uid is None:
            # compute similarity of the full text to each candidate (we'll take top-2 uids)
            sims_full = util.cos_sim(user_emb, cand_embs)[0]
            # aggregate scores per uid (max over its candidate texts)
            uid_scores = {}
            for s, uid in zip(sims_full, cand_index_to_uid):
                s_v = float(s)
                uid_scores[uid] = max(uid_scores.get(uid, -1.0), s_v)
            # sort uids by score
            sorted_uids = sorted(uid_scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_uids) >= 1 and pick_uid is None:
                pick_uid = sorted_uids[0][0]
                debug_scores['fallback_pick_score'] = float(sorted_uids[0][1])
            if len(sorted_uids) >= 2 and place_uid is None:
                place_uid = sorted_uids[1][0]
                debug_scores['fallback_place_score'] = float(sorted_uids[1][1])
            elif place_uid is None and len(sorted_uids) == 1:
                # only one object visible; place location will be a default target
                place_uid = None

        return pick_uid, place_uid, debug_scores


# --------------------------- Controller (simple IK + grasp) ---------------------------

class SimpleController:
    def __init__(self, robot_id):
        self.robot = robot_id
        self.ee_index = EE_LINK_INDEX

    def move_to(self, target_pos, target_ori=None, steps=120):
        if target_ori is None:
            target_ori = p.getQuaternionFromEuler([0, 3.14, 0])
        for i in range(steps):
            joint_poses = p.calculateInverseKinematics(self.robot, self.ee_index, target_pos, target_ori)
            num_joints = p.getNumJoints(self.robot)
            # apply joint poses to the actuated joints only
            for j in range(num_joints):
                # skip fixed joints
                info = p.getJointInfo(self.robot, j)
                joint_type = info[2]
                if joint_type == p.JOINT_FIXED:
                    continue
                # clamp index safety
                try:
                    p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, joint_poses[j], force=200)
                except Exception:
                    pass
            p.stepSimulation()
            time.sleep(1.0/240.0)

    def pick(self, obj_id):
        # Approach above the object
        pos, orn = p.getBasePositionAndOrientation(obj_id)
        pre_pos = [pos[0], pos[1], pos[2] + 0.18]
        grasp_pos = [pos[0], pos[1], pos[2] + 0.02]
        self.move_to(pre_pos, steps=80)
        self.move_to(grasp_pos, steps=80)
        # Create fixed constraint between ee link and object base
        cid = p.createConstraint(self.robot, self.ee_index, obj_id, -1, p.JOINT_FIXED,
                                 [0,0,0], [0,0,0], [0,0,0])
        # Lift
        lift_pos = [pos[0], pos[1], pos[2] + 0.25]
        self.move_to(lift_pos, steps=80)
        return cid

    def place_and_release(self, drop_pos, constraint_id):
        # Move above drop
        drop_above = [drop_pos[0], drop_pos[1], drop_pos[2] + 0.25]
        self.move_to(drop_above, steps=80)
        self.move_to([drop_pos[0], drop_pos[1], drop_pos[2] + 0.05], steps=60)
        # release
        if constraint_id is not None:
            try:
                p.removeConstraint(constraint_id)
            except Exception:
                pass
        # move away
        self.move_to([drop_pos[0], drop_pos[1], drop_pos[2] + 0.3], steps=80)


# --------------------------- Main interactive loop ---------------------------

def main():
    connect_pybullet(gui=True)
    robot, cubes = create_scene()
    controller = SimpleController(robot)
    matcher = SemanticMatcher()

    print('\n=== RT-2 Style Simulation (starter) ===')
    print('Type natural language commands like:')
    print("  pick up the red cube and place it on the blue cube")
    print("  put the blue block onto the green block")
    print("Type 'exit' to quit.\n")

    while True:
        # Step the sim a few times to ensure stable rendering
        for _ in range(15):
            p.stepSimulation()
            time.sleep(1.0/240.0)

        # capture camera & segmentation
        img, seg = get_camera_image_and_seg()
        visible_uids = [u for u in cubes.keys() if (seg == u).any()]
        objects_meta = {u: cubes[u] for u in visible_uids}

        # Print visible objects
        print('Visible objects:')
        for u, m in objects_meta.items():
            pos, _ = p.getBasePositionAndOrientation(u)
            print(f"  uid={u} name={m['name']} pos={tuple(round(x,3) for x in pos)}")

        user = input('\nEnter command: ').strip()
        if user.lower() in ('exit','quit'):
            print('Exiting...')
            break
        if len(user) == 0:
            continue

        # Semantic decide pick/place
        pick_uid, place_uid, dbg = matcher.pick_and_place_from_text(user, objects_meta)
        print('\n[DEBUG] semantic decision:', dbg)

        if pick_uid is None:
            print('Could not decide what to pick. Try a simpler command (e.g., "pick the red cube").')
            continue

        print(f'>> Decided to pick uid={pick_uid} ("{objects_meta[pick_uid]["name"]}")')
        if place_uid is not None:
            print(f'>> Decided to place on uid={place_uid} ("{objects_meta[place_uid]["name"]}")')
        else:
            print('>> No target object found for placement; using default drop location.')

        # Execute pick
        cid = controller.pick(pick_uid)

        # Decide drop location
        if place_uid is not None:
            tgt_pos, _ = p.getBasePositionAndOrientation(place_uid)
            drop_pos = [tgt_pos[0], tgt_pos[1], tgt_pos[2] + 0.06]
        else:
            drop_pos = [0.4, 0.0, 0.05]

        controller.place_and_release(drop_pos, cid)

    p.disconnect()


if __name__ == '__main__':
    main()
