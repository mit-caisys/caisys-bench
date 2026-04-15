# Video-QA Workflow 

Video-QA is a workflow involving video-processing that works on the Video-MME dataset using Langgraph.

##  Getting Started

Follow these instructions to get the Video-QA workflow running on your machine.

1. Make sure all libraries in used for Video-QA are installed using
    ```bash
    video_qa_requirements.sh
    ```
2. Install videos from the dataset using
    ```bash
    download_hugging_face.sh
    ```

3. Start VLLM using the following command where device are the id of gpus (comma seperated), next is model name, tensor parallelism and cpus alloted. For example:
    ```bash
    start_instance/start_vllm.sh　'\"device=0\"' 'google/gemma-3-27b-it' 1 all
    ```
4. Start Whisper using where 1 can also be a comma seperated list of gpus
    ```bash
    start_instance/start_whisper.sh gpu 1
    ```
    or depending if you want to run Whisper on cpu
    ```bash
    start_instance/start_whisper.sh cpu
    ```
5. Start a run of experiments using
    ```bash
    video-qa/run.sh
    ```

You can edit which experiments are run in ```run_experiments.py``` chainging the number of frames and whether STT is used or not.


For the full frequency and qps sweep experiment you can run using 
```bash
video-qa/freq_qps_run.sh
```
Feel free to update the frequencies and qps based on your hardware and preferences.

For the cache experiment you can run using
```bash
video-qa/cache_experiment_run.sh
```

For both the frequency and cache sweep experiments make sure that only one frame number is set in ```run_experiments.py``` and that transcription is set to just True.


