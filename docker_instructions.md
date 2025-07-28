We should have the code, and the docker file in the repo, and if we are using a model.
that model should be present in the repo, if the model is available on internet , that is also good.
we would have internet while building the docker image

but the docker container while running will not have internet connection.

**They said no internet connection atleast for round 1A, that means can we use internet for round 1B?**

the input and output directory would be defined in the build command

like: (pwd) output , (pwd) input.

Build command :
docker build --platform linux/amd64 -t >mysolutionname.someidentifier> .

We will have internet while building the image, but not while running the image.
that means from inside the container there will be no internet access.

### Run command :

docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output:/app/output/repoidentifier/:/app/output --network none <mysolutionname.someidentifier>

when they run these commands, we need to automatically read all the input pdfs,
and give all <filename>.json in the output folder.
