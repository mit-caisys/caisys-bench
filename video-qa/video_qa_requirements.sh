pip install openai langchain-core langchain-openai langgraph pydub datasets requests opencv-python pillow scenedetect langchain matplotlib statsmodels numpy pandas pydantic scipy seaborn PyYAML


pushd /tmp

ffmpeg_dirname="ffmpeg-master-latest-linux64-gpl-shared"
wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${ffmpeg_dirname}.tar.xz
tar -xf ${ffmpeg_dirname}.tar.xz

sudo cp -a ${ffmpeg_dirname}/bin/* /usr/local/bin/

sudo cp -a ${ffmpeg_dirname}/lib/* /usr/local/lib/

sudo ldconfig

rm -rf ${ffmpeg_dirname} ${ffmpeg_dirname}.tar.xz

popd