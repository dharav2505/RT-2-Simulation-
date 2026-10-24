import streamlit as st
from sim_env import SimulationEnvironment
from perception import CLIPPerception
from controller import RobotController
from PIL import Image

st.title("🤖 RT-2 PyBullet Simulation Demo")

env = SimulationEnvironment()
perception = CLIPPerception()
controller = RobotController(env)

st.sidebar.header("Command Input")
command = st.sidebar.text_input("Enter a command:", "pick up the red cube")

if st.sidebar.button("Execute"):
    image = Image.fromarray(env.get_camera_image())
    st.image(image, caption="Camera View", use_column_width=True)
    detected_obj = perception.match_command_to_object(image, command, ["red", "green", "blue"])
    st.write(f"🎯 Matched Object: **{detected_obj}**")
    controller.pick_and_place(detected_obj)
    st.success(f"Executed: {command}")
