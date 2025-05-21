# Heuristics using Machine Learning for GraalVM Native Image

This repository aims to introduce a training pipeline for inlining models based on [Neuroevolution of Augmenting Topologies](en.wikipedia.org/wiki/Neuroevolution_of_augmenting_topologies).  

📝 The OS used for development was `Arch Linux`, kernel `6.14.2`.  
📝 The Python version used was 3.11, dependecies are listed in `requirements.txt`.  
📝 You need to install [mx](github.com/graalvm/mx), which is a utility used to build and benchmark GraalVM.  
📝The Labs JDK version used was `labsjdk-ce-latest-23+24-jvmci-b01`.  

To compile GraalVM on Linux, I've had to change libfffi version from 3.4.4 to 3.4.6 in `truffle/mx.truffle/mx_truffle.py` and `truffle/mx.truffle/suite.py`.  
## Build
Before compiling GraalVM, you dowload a LabsJDK for building Graal:
```bash
mx fetch-jdk
```
Once the JDK is downloaded, you need to copy `.env_template` into `.env` and replace `USE_GRAPHS` with `True` or `False` and `JAVA_HOME`, `GRAAL_REPO_DIR` with absolute paths to the directories.  
If you set `USE_GRAPHS` to `False`, comment out line 429 in `graal/substratevm/src/com.oracle.graal.pointsto/src/com/oracle/graal/pointsto/phases/InlineBeforeAnalysis.java` and uncomment line 431.  

You need to deploy a dummy network for compilation:
```bash
python api.py
```

While the dummy network is deployed, you need to compile GraalVM by running the following command in the `graal/vm` folder in a parallel process:
```bash
mx --java-home <java-path> --env ni-ce build
```

Once the compilation completes, you can kill the dummy network deployed via `api.py` by pressing `Ctrl+C` or killing the process via a process manager.  
## Training

You can launch the training by running:
```bash
python evolve.py
```

## Benchmarking
Once training is done, you can write your own JSON network configuration in the same format as examples found under `configs` folder.  

You then need to call `infere.py` to deploy your network for you:
```bash
python infere.py configs/myconfig.json
```

In a different thread, you may want to launch a benchmark, i.e. `philosophers`:
```
mx --java-home <java-path> --env ni-ce benchmark "rennaisance-native-image:philosophers" -- --jvm=native-image --jvm-config=default-ce
```

Results of the benchmark are published into the file `bench-results.json` in the current working directory.  
## Minimal build for generating a native image
Similarly to benchmarks, you deploy the network first:
```bash
python infere.py configs/myconfig.json
```

If you intend to only manually generate native images and not train or benchmark, you can run a minimal build in the local directories `graal/vm` and then `graal/substratevm`:
```
mx --java-home <java-path> build
```

Then, you compile the Java source code into a Java bytecode:
```
<java-path>/bin/javac -d build src/com/example/HelloWorld.java
```

Now you copy the resulting `HelloWorld.class` from `build` over into `graal/substratevm`, where you run:
```
mx --java-home <java-path> native-image com.example.HelloWorld
```

The resulting native image will be saved in the local working directory under  `helloworld`.  