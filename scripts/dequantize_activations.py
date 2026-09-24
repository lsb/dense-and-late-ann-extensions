#!/usr/bin/env python3
"""Rewrite a dynamically quantized ONNX model into a weight-only int8 model.

The published int8 exports of both encoders use onnxruntime's *dynamic*
quantization: every MatMul becomes

    DynamicQuantizeLinear(x) -> MatMulInteger(x_q, W_q) -> Cast -> Mul(x_scale * W_scale)

with uint8 activations and int8 weights (MiniLM's export also quantizes the
attention products, where both operands are activations). Two problems follow:

* On x86 CPUs without AVX-512 VNNI / AVX-VNNI (for example AMD Zen 3, as on
  GitHub's runners), onnxruntime's u8s8 kernel accumulates pairs of products in
  saturating 16-bit integers. For LateOn-Code-edge this destroys the output
  (per-token cosine 0.2-0.4 against a VNNI machine).
* The activation scale is computed over the whole batch tensor, so a text's
  embedding depends on the other texts in its batch.

This script replaces each such chain with

    MatMul(x, DequantizeLinear(W_q, W_scale, W_zero_point))

(or a plain float MatMul when both operands are activations).

The weights stay int8 in the file (same size); onnxruntime constant-folds the
DequantizeLinear at session creation and computes the products in float32.
The result no longer depends on the CPU or the batch.

Usage: dequantize_activations.py in.onnx out.onnx
"""
import sys
import onnx
from onnx import helper


def rewrite(model):
    g = model.graph
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    inits = {i.name for i in g.initializer}
    drop, new = set(), []
    for mm in [n for n in g.node if n.op_type == "MatMulInteger"]:
        (cast,) = cons[mm.output[0]]
        assert cast.op_type == "Cast"
        (mul,) = cons[cast.output[0]]
        assert mul.op_type == "Mul"
        scales = prod[[i for i in mul.input if i != cast.output[0]][0]]
        assert scales.op_type == "Mul"
        floats = []
        for q, zp in ((mm.input[0], mm.input[2]), (mm.input[1], mm.input[3])):
            if q in inits:
                # A stored int8 weight: dequantize it once, at session creation.
                assert zp in inits
                (scale,) = [i for i in scales.input if i in inits]
                f = q + "_dequantized"
                new.append(helper.make_node("DequantizeLinear", [q, scale, zp], [f], name=f))
                floats.append(f)
            else:
                # A dynamically quantized activation: use the float tensor it came from.
                dql = prod[q]
                assert dql.op_type == "DynamicQuantizeLinear", dql.op_type
                floats.append(dql.input[0])
        new.append(helper.make_node("MatMul", floats, [mul.output[0]], name=mul.output[0] + "_float"))
        drop.update([id(mm), id(cast), id(mul)])
    kept = [n for n in g.node if id(n) not in drop] + new
    # Remove nodes whose outputs are no longer used (the DynamicQuantizeLinear
    # and scale-product nodes), repeating until nothing changes.
    outputs = {o.name for o in g.output}
    while True:
        used = {i for n in kept for i in n.input} | outputs
        alive = [n for n in kept if any(o in used for o in n.output)]
        if len(alive) == len(kept):
            break
        kept = alive
    del g.node[:]
    g.node.extend(kept)
    # Topologically sort, since the new nodes were appended at the end.
    ready = {i.name for i in g.input} | inits | {""}
    order, pending = [], list(g.node)
    while pending:
        rest = []
        for n in pending:
            if all(i in ready for i in n.input):
                order.append(n); ready.update(n.output)
            else:
                rest.append(n)
        assert len(rest) < len(pending), "cycle"
        pending = rest
    del g.node[:]
    g.node.extend(order)
    return model


def main():
    m = onnx.load(sys.argv[1])
    n0 = sum(n.op_type == "MatMulInteger" for n in m.graph.node)
    m = rewrite(m)
    onnx.checker.check_model(m)
    onnx.save(m, sys.argv[2])
    print(f"{sys.argv[1]}: rewrote {n0} MatMulInteger chains -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
