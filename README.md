# Tiny Tensor Compiler

A lightweight compiler for PyTorch inference graphs, built to explore graph lowering, intermediate representations, compiler optimizations, and execution efficiency for machine learning workloads.

The project takes PyTorch models, captures their computation graphs using `torch.fx`, lowers them into a custom intermediate representation, applies compiler optimization passes, and will ultimately benchmark optimized execution against the original model.

## Compiler Pipeline

```text
PyTorch Model
      ↓
torch.fx Graph Capture
      ↓
Compiler Frontend
      ↓
Custom Intermediate Representation
      ↓
Optimization Passes
      ↓
Optimized Graph
      ↓
Execution / Benchmarking
```

## Current Features

* PyTorch graph capture using `torch.fx`
* Custom compiler intermediate representation
* FX-to-IR lowering frontend
* Explicit representation of operations and data dependencies

For example, a PyTorch model containing:

```python
x = self.linear(x)
x = torch.relu(x)
return x
```

is lowered into:

```text
x = input()
linear = linear(x)
relu = relu(linear)
output = return(relu)
```

## Planned Compiler Optimizations

The compiler will progressively support optimization passes including:

* Constant folding and propagation
* Dead code elimination
* Algebraic simplification
* Operator fusion
* Graph canonicalization
* Memory reuse and intermediate tensor lifetime optimization

## Planned Backend Work

Future stages of the project will explore:

* Executing optimized IR graphs
* Benchmarking optimized and baseline PyTorch models
* Measuring inference latency and memory usage
* Kernel-level optimization
* Triton or lower-level code generation for selected tensor operations

## Project Goals

Tiny Tensor Compiler is designed as an end-to-end exploration of modern machine learning compiler systems. Rather than relying entirely on existing compiler abstractions, the project implements core pieces of the compilation pipeline directly, including graph lowering, IR design, optimization passes, and eventually backend execution.

The long-term goal is to develop a small but measurable ML compiler capable of transforming neural network computation graphs and demonstrating concrete performance improvements.

## Repository Structure

```text
tiny-tensor-compiler/
├── compiler/
│   ├── __init__.py
│   ├── frontend.py
│   └── ir.py
├── examples/
│   ├── inspect_graph.py
│   ├── test_ir.py
│   └── compile_model.py
├── README.md
└── requirements.txt
```

## Status

Currently implemented:

```text
PyTorch → FX Graph → Custom IR
```

Next milestone:

```text
Custom IR → Optimization Passes → Optimized IR
```
