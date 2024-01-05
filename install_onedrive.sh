cd ~/Downloads
sudo apt update -y
sudo apt remove -y onedrive
sudo rm -rf /var/lib/dpkg/lock-frontend &&  sudo rm -rf /var/lib/dpkg/lock
wget -qO - https://download.opensuse.org/repositories/home:/npreining:/debian-ubuntu-onedrive/xUbuntu_20.04/Release.key | sudo apt-key add -
echo 'deb https://download.opensuse.org/repositories/home:/npreining:/debian-ubuntu-onedrive/xUbuntu_20.04/ ./' | sudo tee /etc/apt/sources.list.d/onedrive.list
sudo apt update -y
sudo apt-cache search onedrive
sudo apt install onedrive -y
onedrive --version