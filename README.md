На windows. 
python 3.10 Установка
Обязательно установить CUDA SDK 12.8 и Microsoft BuildKit (установить компонент для C++ разработки)
В переменные среды windows добавить переменную CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
1. cmd
- python -m venv ./venv в папке проекта в терминале
- ./venv/Scripts/activate
- pip install uv ninja
- uv pip install -r requirements.txt
2. cmd
  установка torch2trt
- git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
- cd torch2trt
- uv pip install /requirements/requirements_10.txt
- python setup.py install
- cd ..
  установка trt_pose
- git clone https://github.com/NVIDIA-AI-IOT/trt_pose.git
- cd trt_pose
- python setup.py install
- cd ..
Запуск
- python app.py
