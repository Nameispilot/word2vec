package main

import (
	"fmt"
	"log"
	"word2vec"
)

func main(){
	var err error
	var inputs string
	var outputs []string
	var labels []float64

	//getting dataset from .csv
	if inputs,outputs, labels, err = word2vec.MakeInputs("cmd/dataset.csv"); err != nil {
		log.Fatalf("%+v", err)
	}
	fmt.Println(inputs, outputs, labels)
}
