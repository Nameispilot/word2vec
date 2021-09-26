package word2vec

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var epochs = 500
var embeddingsSize = 8
var vocabularySize = 233
var batchSize = 3

//embeddinf, flatten and softmax layers
type nn struct {
	g          *gorgonia.ExprGraph
	w0, w1, w2 *gorgonia.Node
	out        *gorgonia.Node
	predVal    gorgonia.Value
}

func (m *nn) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1}
}

func newNN(g *gorgonia.ExprGraph) *nn {

	//embeddings
	w0 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(vocabularySize, embeddingsSize),
		gorgonia.WithName("embedding"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	//flatten weights CHECK DIMENSIONS!
	w1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(embeddingsSize, 3),
		gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	//softmax layer
	w2 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(3, 3),
		gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	return &nn{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
	}
}

func (m *nn) fwd(x *gorgonia.Node) (err error) {
	var l0, l1, l2 *gorgonia.Node
	var l0dot, l1dot *gorgonia.Node

	//embedding layer
	l0 = x

	if l0dot, err = gorgonia.Mul(l0, m.w0); err != nil {
		return errors.Wrap(err, "Unable to make an embedding layer!")
	}
	l1, _ = gorgonia.Sigmoid(l0dot)

	//flatten layer
	t := tensor.Shape{1, l1.DataSize()}
	newL1, _ := gorgonia.Reshape(l1, t)
	fmt.Println(newL1)

	if l1dot, err = gorgonia.Mul(newL1, m.w1); err != nil {
		return errors.Wrap(err, "Unable to make a flatten layer!")
	}
	l2, _ = gorgonia.Sigmoid(l1dot)

	//softmax layer
	var out *gorgonia.Node
	if out, err = gorgonia.Mul(l2, m.w2); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w2")
	}

	m.out, err = gorgonia.SoftMax(out)
	gorgonia.Read(m.out, &m.predVal)
	return
}

func Word2Vec(inputs int, outputs, targets []int) error {
	var err error
	g := gorgonia.NewGraph()

	//input node
	x := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(embeddingsSize, 1),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)), gorgonia.WithName("x"))

	//output node
	y := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(len(targets), 1), gorgonia.WithName("y"))

	fmt.Println(x.Shape(), y.Shape())

	m := newNN(g)
	if err = m.fwd(x); err != nil {
		return errors.Wrap(err, "Unable to fwd!")
	}

	losses, err := gorgonia.HadamardProd(m.out, y)
	if err != nil {
		return errors.Wrap(err, "Error!")
	}
	cost, _ := gorgonia.Mean(losses)
	if _, err = gorgonia.Grad(cost, m.learnables()...); err != nil {
		return errors.Wrap(err, "Unable to grad!")
	}

	/*vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...))
	//solver := gorgonia.NewAdamSolver()

	//numExamples := len(outputs)
	//batches := numExamples / batchSize

	for i := 0; i < epochs; i++ {
		for b := 0; b < batches; b++ {
			start := b * batchSize
			end := start + batchSize

			/*var yVal tensor.Tensor
			if yVal, err = outT.Slice(MakeRS(start, end)); err != nil {
				return errors.Wrap(err, "Unable to slice outputs!")
			}
			fmt.Println(yVal.Data(), end)

			vm.Reset()
			if err = vm.RunAll(); err != nil {
				return errors.Wrap(err, "Error while training!")
			}
			solver.Step(gorgonia.NodesToValueGrads(m.learnables()))

		}
	} */

	return nil
}
