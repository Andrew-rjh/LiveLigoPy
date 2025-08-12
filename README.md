# LiveLigoPy
실시간 시스템 사운드 번역 & 실시간 LLM 정보 제공 프롬프터

## Installation

```bash
pip install -r requirements.txt
```

필수 의존성(`imgui`, `glfw`, `soundcard` 등)을 설치합니다. Windows 환경에서도 GUI 모드를 실행할 수 있습니다.

## Usage

```bash
python main.py <server-host>
```

기본적으로 ImGui 기반의 오버레이 GUI와 오디오 클라이언트가 동시에 실행됩니다.
`Insert` 키로 GUI 표시를 토글할 수 있습니다.
