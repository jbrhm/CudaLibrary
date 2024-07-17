sudo -v
cmake . -B build
cmake --build build
cmake --install build
cd package/
python3 -m build
cd dist
sudo pip install --force-reinstall cupybara-1.0.2-py3-none-any.whl
