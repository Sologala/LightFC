docker build -t light_fc_ptq .

docker run --rm -v "$(pwd):/host" light_fc_ptq sh -c "cp -r /x5m/model_output /host/"
docker run --rm -v ".:/host" light_fc_ptq "echo hhh"
