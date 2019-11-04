# filename: uninstall.sh
# functie: deinstall conda python3 installatie onder linux ubuntu 18.04 
# Documentatie https://docs.anaconda.com/anaconda/install/uninstall/

conda install anaconda-clean
anaconda-clean

# ga naar home directory
cd

rm -rf ~/anaconda3

echo "edit ~/.bash_profile remove conda path"

echo "edit ~/.bashrc remove conda "