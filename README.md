# How to run

## Using Linux
1. Install Julia, I used `Julia 1.11.0-DEV.1141` (nightly at the time of writing) for testing. More recent versions might work as well.
2. Clone this repository and check out the `blog` branch.
3. Start the Julia REPL and instantiate the environment using the package mode. This will a fork install MLIR.jl that includes a recent MLIR artifact.
  ```
  > ] # to enter package mode
  > activate .
  > instantiate
  ```
4. run `main.jl`
  ```
> <backspace> # to leave package mode
> include("./main.jl")
  ```

## Using another operating system
This is untested and regular project instantiation will not work because no artifacts for the OS are provided. If you build MLIR from source or get your hands on a release yourself, you could try `dev` the `MLIR.jl` package and change [these lines](https://github.com/jumerckx/MLIR.jl/blob/4237533ec17aa59eb707c41e297fa516a88b3d28/src/MLIR.jl#L15C1-L17C94) to point to the correct location of your install.
