git submodule add -b stable https://github.com/pybind/pybind11.git extern/pybind11

wget https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu126.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu126.zip

within build dir
```
cmake -DPython_EXECUTABLE=/home/ubuntu/pripri-labs/cs599k/.venv/bin/python ..
```

installing site-packages
```
make && cmake --install .
```