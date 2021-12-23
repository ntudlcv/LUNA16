cd data
wget https://zenodo.org/record/3723295/files/sampleSubmission.csv?download=1 -O sampleSubmission.csv
wget https://zenodo.org/record/3723295/files/annotations.csv?download=1 -O annotations.csv
wget https://zenodo.org/record/3723295/files/subset0.zip?download=1 -O subset0.zip
wget https://zenodo.org/record/3723295/files/subset1.zip?download=1 -O subset1.zip
wget https://zenodo.org/record/3723295/files/subset2.zip?download=1 -O subset2.zip
wget https://zenodo.org/record/3723295/files/subset3.zip?download=1 -O subset3.zip
# wget https://zenodo.org/record/3723295/files/subset4.zip?download=1 -O subset4.zip
# wget https://zenodo.org/record/3723295/files/subset5.zip?download=1 -O subset5.zip
# wget https://zenodo.org/record/3723295/files/subset6.zip?download=1 -O subset6.zip
# wget https://zenodo.org/record/4121926/files/subset7.zip?download=1 -O subset7.zip
# wget https://zenodo.org/record/4121926/files/subset8.zip?download=1 -O subset8.zip
# wget https://zenodo.org/record/4121926/files/subset9.zip?download=1 -O subset9.zip
unzip evaluationScript.zip
zip -FF subset0.zip -O subset0.fixed.zip && unzip subset0.fixed.zip
zip -FF subset1.zip -O subset1.fixed.zip && unzip subset1.fixed.zip
zip -FF subset2.zip -O subset2.fixed.zip && unzip subset2.fixed.zip
zip -FF subset3.zip -O subset3.fixed.zip && unzip subset3.fixed.zip
# zip -FF subset4.zip -O subset4.fixed.zip && unzip subset4.fixed.zip
# zip -FF subset5.zip -O subset5.fixed.zip && unzip subset5.fixed.zip
# zip -FF subset6.zip -O subset6.fixed.zip && unzip subset6.fixed.zip
# zip -FF subset7.zip -O subset7.fixed.zip && unzip subset7.fixed.zip
# zip -FF subset8.zip -O subset8.fixed.zip && unzip subset8.fixed.zip
# zip -FF subset9.zip -O subset9.fixed.zip && unzip subset9.fixed.zip
mkdir train
mkdir test
mv subset0/* test/
mv subset*/* train/
rm *.zip
rm -r -f subset*
cd ..