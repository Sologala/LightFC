FROM openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8-py310

WORKDIR /x5m

COPY . .


RUN hb_mapper makertbin --fast-perf --model-type onnx --model ./lightfc_detection.onnx
RUN cd model_output && hb_perf lightfc_detection.bin

