package word2vec

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var epochs = 100
var embeddingsSize = 8
var vocabularySize = 233

//embedding and softmax layers
type nn struct {
	g       *gorgonia.ExprGraph
	w0, w1  *gorgonia.Node
	out     *gorgonia.Node
	predVal gorgonia.Value
}

func (m *nn) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1}
}

func newNN(g *gorgonia.ExprGraph, vocab int) *nn {

	//embeddings
	w0 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(vocabularySize, embeddingsSize),
		gorgonia.WithName("embedding"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	//linear layer
	w1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(vocabularySize, vocab),
		gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	return &nn{
		g:  g,
		w0: w0,
		w1: w1,
	}
}

func (m *nn) fwd(x *gorgonia.Node) (err error) {
	var l0, l1, l2 *gorgonia.Node
	var l0dot, l1dot *gorgonia.Node

	//embedding layer
	l0 = x

	wTrans, _ := gorgonia.Transpose(m.w0)

	if l0dot, err = gorgonia.Mul(l0, wTrans); err != nil {
		return errors.Wrap(err, "Unable to make an embedding layer!")
	}
	l1, _ = gorgonia.Sigmoid(l0dot)

	//linear layer
	if l1dot, err = gorgonia.Mul(l1, m.w1); err != nil {
		return errors.Wrap(err, "Unable to make a softmax layer!")
	}
	l2, _ = gorgonia.Sigmoid(l1dot)

	m.out = l2
	gorgonia.Read(m.out, &m.predVal)
	return nil
}

func Word2Vec(inputs int, outputs []int, targets []float64) error {
	var err error
	g := gorgonia.NewGraph()

	//input node
	x := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, embeddingsSize),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)), gorgonia.WithName("x"))

	//output node
	y := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, len(targets)), gorgonia.WithName("y"))
	yT := tensor.New(tensor.WithBacking(targets))
	gorgonia.Let(y, yT)

	m := newNN(g, len(targets))
	if err = m.fwd(x); err != nil {
		return errors.Wrap(err, "Unable to fwd!")
	}

	/*losses, err := gorgonia.HadamardProd(m.out, y)
	if err != nil {
		return errors.Wrap(err, "Error!")
	}
	cost, _ := gorgonia.Mean(losses)
	cost, _ = gorgonia.Neg(cost) */

	losses, _ := gorgonia.Sub(m.out, y)
	square, _ := gorgonia.Square(losses)
	cost, _ := gorgonia.Mean(square)

	if _, err = gorgonia.Grad(cost, m.learnables()...); err != nil {
		return errors.Wrap(err, "Unable to grad!")
	}

	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...))
	solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.01))

	for i := 0; i < epochs; i++ {
		vm.Reset()
		if err = vm.RunAll(); err != nil {
			return errors.Wrap(err, "Error while training!")
		}
		solver.Step(gorgonia.NodesToValueGrads(m.learnables()))
	}
	tens := tensor.New(tensor.WithBacking(m.predVal.Data()))
	for i := 0; i < tens.Cap(); i += 3 {
		tmp := targets[i : i+3]
		V, _ := tens.Slice(MakeRS(i, i+3))
		fmt.Println(tmp)
		fmt.Printf("%.2f\n", V)
	}

	return nil
}
