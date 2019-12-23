# Datasets Preprocessing

## Install Dependecies

```bash
./install_deps.sh
```

## Usage

```bash
make <dataset>
./<dataset> m 
```

`<dataset>`: audio|glove|mnist|enron
`m`: dimension for Hamming data (suggested value: rounding original dim to multiples of 64)

SIFT|GIST|SIFT1B|GIST80M will be added soon