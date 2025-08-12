import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
import OpenGL.GL as gl  # 선택: 화면 클리어 용

def run_ui() -> None:
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")

    window = glfw.create_window(800, 600, "LiveLingo", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)

    # ★ ImGui 컨텍스트 생성 (중요)
    imgui.create_context()

    impl = GlfwRenderer(window)

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            impl.process_inputs()

            imgui.new_frame()

            imgui.begin("LiveLingo")
            imgui.text("ImGui UI placeholder")
            imgui.end()

            # 선택: 화면 클리어
            gl.glClearColor(0.1, 0.1, 0.1, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            imgui.render()
            impl.render(imgui.get_draw_data())
            glfw.swap_buffers(window)
    finally:
        impl.shutdown()
        imgui.destroy_context()   # ★ 깔끔 종료
        glfw.terminate()
