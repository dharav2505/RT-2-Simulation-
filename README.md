# 🤖 RT-2-Inspired Language-Conditioned Pick-and-Place

> Type a command in plain English — "pick up the red cube and place it on the blue cube" — and watch a simulated KUKA iiwa arm carry it out in PyBullet.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Simulator](https://img.shields.io/badge/simulator-PyBullet-orange)
![NLP](https://img.shields.io/badge/embeddings-sentence--transformers-green)

This project is a lightweight, single-file starter inspired by the **Vision-Language-Action (VLA)** idea behind [RT-2](https://arxiv.org/abs/2307.15818): use pretrained language representations so robot commands don't have to match a fixed vocabulary.

> **Note:** This is *RT-2-inspired*, not an RT-2 implementation. There is no VLA transformer, no action tokenization, and no learned control policy. Language grounding uses sentence embeddings; motion uses classical inverse kinematics.

---

## ✨ Key Features

- **Free-form natural-language commands** via console input, with no hardcoded command strings.
- **Semantic object grounding** using `all-MiniLM-L6-v2` sentence embeddings and cosine similarity, so "block", "cube", and "colored cube" all resolve correctly.
- **Pick / place phrase extraction** with regex, plus a whole-sentence semantic fallback when parsing fails.
- **Simulated perception**: RGB + segmentation camera determines which objects are visible.
- **Pick-and-place control** with PyBullet IK, position control, and a fixed-constraint grasp.
- **Inspectable decisions**: similarity scores and chosen object IDs are printed on every command.
- **Modular design**: swap in CLIP, an LLM parser, or a real VLA without touching the controller.

---

## 🏗️ System Architecture

```
                ┌────────────────────────────┐
 User command → │ SemanticMatcher            │
 (console)      │  • regex pick/place split  │
                │  • MiniLM embeddings       │ → (pick_uid, place_uid)
                │  • cosine-sim to objects   │          │
                └────────────▲───────────────┘          ▼
                             │                ┌───────────────────┐
   visible objects           │                │ SimpleController  │
┌────────────────────────┐   │                │  • IK waypoints   │
│ PyBullet scene         │───┘                │  • position ctrl  │
│  plane + iiwa + cubes  │◄───────────────────│  • fixed grasp    │
│  camera + segmentation │   sim step         └───────────────────┘
└────────────────────────┘
```

| Component | Responsibility |
|---|---|
| `create_scene()` | Loads plane, KUKA iiwa, and red/green/blue cubes |
| `get_camera_image_and_seg()` | Renders 320×240 RGB + segmentation mask |
| `seg_to_object_crops()` | Per-object image crops (hook for future CLIP grounding; currently unused) |
| `SemanticMatcher` | Maps command text to pick and place object IDs |
| `SimpleController` | `move_to`, `pick`, `place_and_release` |
| `main()` | Interactive loop: observe → ground → act |

**Pipeline per command:** settle physics → capture camera → list visible cubes → read command → extract phrases → embed and match → pick (pre-grasp, grasp, lift) → place (above, lower, release, retreat).

---

## 📋 Prerequisites & Dependencies

| Requirement | Notes |
|---|---|
| Python | 3.9 or newer recommended |
| OS | Windows, Linux, or macOS with a display (the script opens the PyBullet GUI) |
| GPU | **Not required.** MiniLM runs comfortably on CPU |
| RAM | ~2 GB free |

**Python packages**

```
pybullet
sentence-transformers
numpy
pillow
opencv-python
```

`sentence-transformers` installs PyTorch automatically. The first run downloads the `all-MiniLM-L6-v2` model (~90 MB), so an internet connection is needed once.

---

## 🚀 Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install pybullet sentence-transformers numpy pillow opencv-python
```

**Optional environment variables**

```bash
# Cache the embedding model somewhere specific
export SENTENCE_TRANSFORMERS_HOME=./models        # Windows: set SENTENCE_TRANSFORMERS_HOME=.\models
```

> 💡 On Windows, `pip install pybullet` may need the Microsoft C++ Build Tools if no prebuilt wheel exists for your Python version. Using Python 3.9–3.11 usually avoids this.

---

## ▶️ Usage

```bash
python rt2_sim_starter.py
```

A PyBullet window opens and the console prompts for commands:

```
Visible objects:
  uid=2 name=red   pos=(0.55, -0.15, 0.03)
  uid=3 name=green pos=(0.55, 0.0, 0.03)
  uid=4 name=blue  pos=(0.55, 0.15, 0.03)

Enter command: pick up the red cube and place it on the blue cube
```

**Example commands**

| Command | Behavior |
|---|---|
| `pick up the red cube and place it on the blue cube` | Stacks red on blue |
| `put the blue block onto the green block` | Stacks blue on green |
| `grab the green thing` | Picks green, drops at default location `[0.4, 0, 0.05]` |
| `exit` or `quit` | Closes the simulation |

---

## 📁 Project Directory Structure

```
.
├── rt2_sim_starter.py     # Full simulation (scene, perception, matcher, controller, main loop)
├── README.md              # This file
├── requirements.txt       # pybullet, sentence-transformers, numpy, pillow, opencv-python
└── docs/
    └── technical_report.md  # Project report (optional)
```

Possible future split:

```
rt2_sim/
├── scene.py        # create_scene, camera
├── perception.py   # segmentation, crops, CLIP grounding
├── language.py     # SemanticMatcher / LLM parser
├── control.py      # SimpleController
└── main.py
```

---

## ⚠️ Known Limitations

- Object names come from simulator metadata, not from the image pixels.
- Execution is open-loop: no re-grasp or failure recovery.
- Grasping uses a rigid constraint instead of gripper contact physics.
- Regex parsing supports a limited set of verbs and sentence shapes.

## 🛣️ Roadmap

- [ ] CLIP-based visual grounding using `seg_to_object_crops`
- [ ] LLM/VLM structured command parsing
- [ ] Parallel-jaw gripper with real contact
- [ ] Closed-loop pose re-estimation and retries
- [ ] Randomized scenes and an evaluation benchmark
- [ ] Gymnasium wrapper and small VLA fine-tuning on scripted demos

## 📚 References

- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023.
- Reimers & Gurevych, *Sentence-BERT*, EMNLP 2019.
- [PyBullet Quickstart Guide](https://pybullet.org)

## 📄 License

Add your preferred license (e.g., MIT) here.
