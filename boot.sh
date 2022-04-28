#! /bin/sh

# install
sudo apt update
sudo apt upgrade
sudo apt install python-pip
sudo apt install python3-dev
sudo apt install python3.10-venv

# make venv
python3 -m venv env
source env/bin/activate

# install
pip install -r requirements.txt

# pull
git pull

# run
chmod +x app.py
python3 app.py
