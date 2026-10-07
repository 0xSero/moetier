#!/bin/bash
# N124: build _moe_n124.so (a9 + power-of-two GU/DN splits) in image 24c872759256, CPU only, slot B cpuset.
D=$HOME/n124/build
timeout 1500 docker run --rm --name n124-build --network none --memory 32g --memory-swap 32g --cpuset-cpus 44-47 -v $D:/w \
  --entrypoint bash 24c872759256 -c "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; cd /w; \
  T=\$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))'); \
  icpx -fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC -std=c++17 -shared -fsycl-device-code-split=per_kernel \
  -D_GLIBCXX_USE_CXX11_ABI=1 -I csrc -I\$T/include -I\$T/include/torch/csrc/api/include -x c++ csrc/exl3_moe.n124.sycl -x none -o /w/_moe_n124.so \
  -L\$T/lib -Wl,-rpath,\$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu 2>&1 | grep -E 'error|Error' -A3; \
  chown $(id -u):$(id -g) /w/_moe_n124.so; ls -la /w/_moe_n124.so"
echo "build rc=$?"
