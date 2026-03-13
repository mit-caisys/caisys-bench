# Video-QA Workflow 

Video-QA is a workflow involving video-processing that works on the Video-MME dataset using Langgraph.

##  Getting Started

Follow these instructions to get the Video-QA workflow running on your machine.

1. Make sure all libraries used for Video-QA are installed using
    ```bash
    install_libraries.sh
    ```

2. Start VLLM using the following command where device are the id of gpus (comma seperated), next is model name, and finally the tensor parrallelism. For example:
    ```bash
    vllm_start.sh　'\"device=1\"' 'mistralai/Mistral-Small-3.2-24B-Instruct-2506' 1
    ```
3. Start Whisper using where 0 can also be a comma seperated list of gpus
    ```bash
    start_whisper.sh gpu 0
    ```
    or depending if you want to run Whisper on cpu
    ```bash
    start_whisper.sh cpu
    ```

4. Run experiments and generate graph image using
    ```bash
    run.sh
    ```

    The accuarcy graph should look something like the following:

    ![An image of accuracy compared across configurations](./example_plots/accuracies.png)

##  Comparing Results

The CSV files for different configurations we ran are in the public_experiments_results folder. Feel free to run similar or the same configurations to make sure you are obtaining comaprable resutls, or run different configurations to see the effects of changing configurations. 