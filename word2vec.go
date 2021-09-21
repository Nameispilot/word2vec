package word2vec

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var epochs = 500
var embeddingsSize = 8
var vocabularySize = HashBasis

//embeddinf, flatten and softmax layers
type nn struct {
	g          *gorgonia.ExprGraph
	w0, w1, w2 *gorgonia.Node
	out        *gorgonia.Node
	predVal    gorgonia.Value
}

func newNN(g *gorgonia.ExprGraph) *nn {

	//embeddings
	w0 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(vocabularySize, embeddingsSize),
		gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	//flatten weights CHECK DIMENSIONS!
	w1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, embeddingsSize),
		gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	return &nn{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
	}
}

func Word2Vec(inputs int, outputs, targets []int) {

}
