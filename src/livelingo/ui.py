"""Simple ImGui UI placeholder for LiveLingo."""

import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw


def run_ui() -> None:
    """Launch a basic ImGui window displaying placeholder text."""
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")

    window = glfw.create_window(800, 600, "LiveLingo", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    impl = GlfwRenderer(window)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()
        imgui.begin("LiveLingo")
        imgui.text("ImGui UI placeholder")
        imgui.end()
        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()
