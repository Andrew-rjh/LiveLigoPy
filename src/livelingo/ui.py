import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
import OpenGL.GL as gl  # OpenGL bindings for rendering

def run_ui() -> None:
    """Run the ImGui overlay user interface."""
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")

    # Configure window for transparent overlay that stays on top
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)
    glfw.window_hint(glfw.MOUSE_PASSTHROUGH, glfw.TRUE)

    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    width, height = mode.size.width, mode.size.height
    window = glfw.create_window(width, height, "LiveLingo", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.set_window_pos(window, 0, 0)
    glfw.make_context_current(window)

    # ImGui context setup
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Enable alpha blending for transparency
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    show = True
    prev_insert = glfw.RELEASE

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            impl.process_inputs()

            # Toggle UI visibility with the Insert key
            state = glfw.get_key(window, glfw.KEY_INSERT)
            if state == glfw.PRESS and prev_insert == glfw.RELEASE:
                show = not show
            prev_insert = state

            gl.glClearColor(0.0, 0.0, 0.0, 0.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            imgui.new_frame()
            if show:
                imgui.begin("LiveLingo", True, imgui.WINDOW_ALWAYS_AUTO_RESIZE)
                imgui.text("ImGui UI placeholder")
                imgui.end()
            imgui.render()
            if show:
                impl.render(imgui.get_draw_data())
            glfw.swap_buffers(window)
    finally:
        impl.shutdown()
        imgui.destroy_context()
        glfw.terminate()
