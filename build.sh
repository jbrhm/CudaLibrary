sudo -v
cmake . -B build -DCPACK_COMPONENTS_ALL="CUPYBARA_DEPS"
cmake --build build
cd build
sudo cmake --install .
sudo cpack -G DEB 
cd ..
cp build/_packages/cupybara_1.0.2.deb releases
cd package/
python3 -m build
cd dist
pip install --force-reinstall cupybara-1.0.2-py3-none-any.whl
cd ../..
cp package/dist/cupybara-1.0.2-py3-none-any.whl releases/cupybara-1.0.2-py3-none-any.whl
