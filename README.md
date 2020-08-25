# E2H: Euclidean Datasets to Hamming Datasets

E2H implements the preprocessing tool used in our recent paper, 
"Long Gong, Huayi Wang, Mitsunori Ogihara, and Jun Xu. 2020. IDEC: indexable distance estimating codes for approximate nearest neighbor search. <i>Proc. VLDB Endow.</i> 13, 9 (May 2020), 1483–1497. DOI:https://doi.org/10.14778/3397230.3397243." 
E2H is used to convert Euclidean datasets to Hamming datasets. 

## Install Dependecies

```bash
./install_deps.sh
```

## Usage

```bash
make <dataset>
./<dataset> m 
```

`<dataset>`: audio|glove|mnist|enron|sift1m|gist1m|sift1b|gist80m 
`m`: dimension for Hamming data (suggested value: rounding original dim to multiples of 64)
