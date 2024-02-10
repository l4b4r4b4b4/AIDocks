#!/bin/bash
python -m llava.serve.controller --host 0.0.0.0 --port 10000 &
# # Launch a gradio web server
python -m llava.serve.gradio_web_server --controller http://llava:10000 --model-list-mode reload --port 8888 &

# # Launch a model worker
LOAD_4BIT_OPTION=""
if [ "$LOAD_4BIT" = "true" ]; then
    LOAD_4BIT_OPTION="--load-4bit"
fi

python -m llava.serve.model_worker --host 0.0.0.0 --controller http://llava:10000 --port 40000 --worker http://llava:40000 --model-path $MODEL_PATH $LOAD_4BIT_OPTION