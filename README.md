LSALM
=====

Latent semantic analysis language model interpolated with n-grams


# Trainen
## Train SRILM taalmodel
```bash
ngram-count blabla
```

Er moet ook een model voor 5-grammen gemaakt worden.

## LSA bestanden aanmaken

```bash
python createIndex.py -o 1 -m train.mm -d train.dict -l train.files -C train/counts … enzo
```

# Developen
## Alle trigrammen uit een tekst halen

```bash
python createIndex.py -o 2 -e dev-2/dev-2.all.eval -E -l dev.files

python createIndex.py -o 3 -e dev-3/dev-3.all.eval -E -l dev.files

python createIndex.py -o 5 -e dev-5/dev-5.all.eval -E -l dev.files
```

## Alle huge-grammen uit een tekst halen

```bash
python createIndex.py -o n -e dev-n/dev-n.all.eval -E -l dev.files
```

## De huge-grammen voor verschillende contextlengtes

```bash
for csize in 5 10 25 50 100 250 500; do awk -v r=$csize '{if(NF > 2 && NF < (r+1)) {print $0}}' dev-n/dev-n.eval > dev-n/dev-n.c$csize.eval; done
```

## Alle contexten bepalen
```bash
awk '{print $1 "     "}' dev-2/dev-2.all.eval | sort -u > dev-2/dev-N2.all.cxp

awk '{print $1, $2 "     "}' dev-3/dev-3.all.eval | sort -u > dev-3/dev-N3.all.cxp

awk '{print $1 "     " $3}' dev-3/dev-3.all.eval | sort -u > dev-3/dev-C3.all.cxp

awk '{print $1 $2 "     " $4 $5}' dev-5/dev-5.all.eval | sort -u > dev-5/dev-C5.all.cxp

awk '{$(NF-1) = ""; $NF = ""; print}' dev-n/dev-n.all.eval | sed 's/  $/   /' | sed '/^[[:space:]]*$/d' | sort -u > dev-n/dev-R2.all.cxp
```

## Contextbestanden aanmaken
```bash
python createContexts.py -c dev-2/dev-N2.all.cxp -I dev-2/dev-N2.all.idx -C dev-2/N2Contexts -v 20k.vocab -T 30

python createContexts.py -c dev-3/dev-N3.all.cxp -I dev-3/dev-N3.all.idx -C dev-3/N3Contexts -v 20k.vocab -T 30

python createContexts.py -c dev-3/dev-C3.all.cxp -I dev-3/dev-C3.all.idx -C dev-3/C3Contexts -v 20k.vocab -T 30

python createContexts.py -c dev-5/dev-C5.all.cxp -I dev-5/dev-C5.all.idx -C dev-5/C5Contexts -v 20k.vocab -T 28
```

## SRILM kansen bepalen voor de contexten 2 & 3
```bash
for i in `seq 0 $(ls -f -1 dev-2/N2Contexts/ | wc -l)`; do echo $i; done | xargs -P 30 -I {} sh -c 'ngram -ppl dev-2/N2Contexts/context.{} -lm train.lm -debug 2 -no-sos -no-eos | python interpretSRILM.py > dev-2/N2SRILMProbs/context.{} '

for i in `seq 0 $(ls -f -1 dev-3/N3Contexts/ | wc -l)`; do echo $i; done | xargs -P 30 -I {} sh -c 'ngram -ppl dev-3/N3Contexts/context.{} -lm train.lm -debug 2 -no-sos -no-eos | python interpretSRILM.py > dev-3/N3SRILMProbs/context.{} '

for i in `seq 0 $(ls -f -1 dev-3/C3Contexts/ | wc -l)`; do echo $i; done | xargs -P 30 -I {} sh -c 'ngram -ppl dev-3/C3Contexts/context.{} -lm train.lm -debug 2 -no-sos -no-eos | python interpretSRILM.py > dev-3/C3SRILMProbs/context.{} '

for i in `seq 0 $(ls -f -1 dev-5/C5Contexts/ | wc -l)`; do echo $i; done | xargs -P 30 -I {} sh -c 'ngram -ppl dev-5/C5Contexts/context.{} -lm train.lm -debug 2 -no-sos -no-eos | python interpretSRILM.py > dev-3/C5SRILMProbs/context.{} '
```

## LSA-modellen maken

lsaN3 is hier om het even, het kan met elke instantie van LSALM

```bash
for gamma in `seq 0 0.05 2.5`; do for dimensions in 400; do python lsaN3.py -D -k $dimensions -g $gamma -T 28 -m train.mm -d train.dict -v 3 -w lsa/train-$gamma-$dimensions-fc.lsa -C lsa/train-$gamma-$dimensions-fc.wc -L lsa/train-$gamma-$dimensions-fc.lc; done; done
```

## Wat kleinere random dev eval sets aanmaken

```bash
awk '{if(NF > 2) print $0}' dev-n/dev-n.all.eval > dev-n/dev-R2.all.eval

awk '{if(NF > 3) print $0}' dev-n/dev-n.all.eval > dev-n/dev-R3.all.eval

for evalset in `seq 0 10`; do shuf -n 2500 dev-n/dev-R2.all.eval > dev-n/shuf/dev-R2.shuf2500-$evalset.eval; shuf -n 2500 dev-n/dev-R3.all.eval > dev-n/shuf/dev-R3.shuf2500-$evalset.eval; done

for csize in 5 10 25 50 100 250 500; do for evalset in `seq 0 10`; do shuf -n 2500 dev-n/dev-n.c$csize.eval > dev-n/shuf/dev-n.shuf2500-c$csize-$evalset.eval; done; done
```

# Reproductie R3
## Pompen

Meerdere samples, meerdere gammas, en meerdere dimensies

```bash
for evalset in 0; do for gamma in `seq 0.00 0.05 2.5`; do for dimensions in 400; do python lsaR3.py -D -k $dimensions -g $gamma -T 28 -m train.mm -d train.dict -v 1 -r lsa/train-$gamma-$dimensions-fc.lsa -c lsa/train-$gamma-$dimensions-fc.wc -l lsa/train-$gamma-$dimensions-fc.lc -i dev-n/shuf/dev-R3.shuf2500-$evalset.eval -s dev-3/dev-N3.all.idx -S dev-3/N3SRILMProbs -o dev-n/out/dev-R3-$gamma-$dimensions-$evalset-fc.out; done; done; done
``` 

## Laat de perplexiteiten zien
```bash
paste <(for i in `seq 0.00 0.05 2.50`; do echo $i; done) <(for i in `seq 0.00 0.05 2.50`; do cut -f2 dev-n/out/dev-R3-$i-400-0-fc.out | ppl | cut -f2 -d' '; done) <(for i in `seq 0.00 0.05 2.50`; do cut -f3 dev-n/out/dev-R3-$i-400-0-fc.out | ppl | cut -f2 -d' '; done) <(for i in `seq 0.00 0.05 2.50`; do cut -f4 dev-n/out/dev-R3-$i-400-0-fc.out | ppl | cut -f2 -d' '; done)
```

## Klein zijstapje naar lineaire interpolatie
```bash
for evalset in `seq 1 10`; do for gamma in `seq 0.00 0.05 2.50`; do for dimensions in 400; do python lsaR3.py -D -k $dimensions -g $gamma -T 28 -m train.mm -d train.dict -v 1 -r lsa/train-$gamma-$dimensions-fc.lsa -c lsa/train-$gamma-$dimensions-fc.wc -l lsa/train-$gamma-$dimensions-fc.lc -i dev-n/shuf/dev-R3.shuf2500-$evalset.eval -s dev-3/dev-N3.all.idx -S dev-3/N3SRILMProbs -o dev-n/out/dev-R3-$gamma-$dimensions-$evalset-fc.lin.out --linear; done; done; done

paste <(for i in `seq 0.00 0.05 2.50`; do echo $i; done) <(for i in `seq 0.00 0.05 2.50`; do cut -f2 dev-n/out/dev-R3-$i-400-0-fc.lin.out | ppl | cut -f2 -d' '; done) <(for i in `seq 0.00 0.05 2.50`; do cut -f3 dev-n/out/dev-R3-$i-400-0-fc.lin.out | ppl | cut -f2 -d' '; done) <(for i in `seq 0.00 0.05 2.50`; do cut -f4 dev-n/out/dev-R3-$i-400-0-fc.lin.out | ppl | cut -f2 -d' '; done)
```

## Plot de data

Bereken het minimum, maximum, en het gemiddelde voor 10 (0-9) devsetsamples, voor p, pl en pb

```bash
paste <(for i in `seq 0.00 0.05 2.50`; do echo $i; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in `seq 0 9`; do cut -f2 dev-n/out/dev-R3-$i-400-$j-fc.lin.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}'; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in `seq 0 9`; do cut -f3 dev-n/out/dev-R3-$i-400-$j-fc.lin.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in `seq 0 9`; do cut -f4 dev-n/out/dev-R3-$i-400-$j-fc.lin.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) > dev-n/data/minmaxavgLinear.dat

paste <(for i in `seq 0.00 0.05 2.50`; do echo $i; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in `seq 0 9`; do cut -f2 dev-n/out/dev-R3-$i-400-$j-fc.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}'; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in `seq 0 9`; do cut -f3 dev-n/out/dev-R3-$i-400-$j-fc.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in `seq 0 9`; do cut -f4 dev-n/out/dev-R3-$i-400-$j-fc.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) > dev-n/data/minmaxavgGeometric.dat
```

## Plotten die hap
```gnuplot
set logscale y
set xlabel
set yrange [1:100000]

plot "minmaxavgGeometric.dat" u 1:2:3 title "range Geo" w filledcu, "minmaxavgGeometric.dat" u 1:4 title "avgGeo" w l linecolor rgb "black", "minmaxavgLinear.dat" u 1:5:6 title "range lsalm" w filledcu, "minmaxavgLinear.dat" u 1:8:9 title "range ngram" w filledcu, "minmaxavgLinear.dat" u 1:7 title "avg lsalm" w l linecolor rgb "blue", "minmaxavgLinear.dat" u 1:10 title "avg ngram" w l linecolor rgb "blue","minmaxavgLinear.dat" u 1:2:3 title "range Lin" w filledcu, "minmaxavgLinear.dat" u 1:4 title "avgLin" w l linecolor rgb "blue"
```

# Reproductie R2
## Pompen

Meerdere samples, meerdere gammas, en meerdere dimensies

```bash
for evalset in 0; do for gamma in `seq 0.00 0.05 2.5`; do for dimensions in 400; do python lsaR2.py -D -k $dimensions -g $gamma -T 28 -m train.mm -d train.dict -v 1 -r lsa/train-$gamma-$dimensions-fc.lsa -c lsa/train-$gamma-$dimensions-fc.wc -l lsa/train-$gamma-$dimensions-fc.lc -i dev-n/shuf/dev-R2.shuf2500-$evalset.eval -s dev-2/dev-N2.all.idx -S dev-2/N2SRILMProbs -o dev-n/out/dev-R2-$gamma-$dimensions-$evalset-fc.out; done; done; done
```

## Laat de perplexiteiten zien
```bash
paste <(for i in `seq 0.00 0.05 2.50`; do echo $i; done) <(for i in `seq 0.00 0.05 2.50`; do cut -f2 dev-n/out/dev-R2-$i-400-0-fc.out | ppl | cut -f2 -d' '; done) <(for i in `seq 0.00 0.05 2.50`; do cut -f3 dev-n/out/dev-R2-$i-400-0-fc.out | ppl | cut -f2 -d' '; done) <(for i in `seq 0.00 0.05 2.50`; do cut -f4 dev-n/out/dev-R2-$i-400-0-fc.out | ppl | cut -f2 -d' '; done)
```

## Idem voor lineaire interpolatie
```bash
for evalset in 0; do for gamma in `seq 0.00 0.05 2.5`; do for dimensions in 400; do python lsaR2.py -D -k $dimensions -g $gamma -T 28 -m train.mm -d train.dict -v 1 -r lsa/train-$gamma-$dimensions-fc.lsa -c lsa/train-$gamma-$dimensions-fc.wc -l lsa/train-$gamma-$dimensions-fc.lc -i dev-n/shuf/dev-R2.shuf2500-$evalset.eval -s dev-2/dev-N2.all.idx -S dev-2/N2SRILMProbs --linear -o dev-n/out/dev-R2-$gamma-$dimensions-$evalset-fc.lin.out; done; done; done
```

## Lin PPL
```bash
paste <(for i in `seq 0.00 0.05 2.50`; do echo $i; done) <(for i in `seq 0.00 0.05 2.50`; do cut -f2 dev-n/out/dev-R2-$i-400-0-fc.lin.out | ppl | cut -f2 -d' '; done) <(for i in `seq 0.00 0.05 2.50`; do cut -f3 dev-n/out/dev-R2-$i-400-0-fc.lin.out | ppl | cut -f2 -d' '; done) <(for i in `seq 0.00 0.05 2.50`; do cut -f4 dev-n/out/dev-R2-$i-400-0-fc.lin.out | ppl | cut -f2 -d' '; done)
```

Bereken het minimum, maximum, en het gemiddelde voor 1 devsetsample, voor p, pl en pb
```bash
paste <(for i in `seq 0.00 0.05 2.50`; do echo $i; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in 0; do cut -f2 dev-n/out/dev-R2-$i-400-$j-fc.lin.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}'; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in 0; do cut -f3 dev-n/out/dev-R2-$i-400-$j-fc.lin.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in 0; do cut -f4 dev-n/out/dev-R2-$i-400-$j-fc.lin.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) > dev-n/data/minmaxavgLinear.R2.dat

paste <(for i in `seq 0.00 0.05 2.50`; do echo $i; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in 0; do cut -f2 dev-n/out/dev-R2-$i-400-$j-fc.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}'; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in 0; do cut -f3 dev-n/out/dev-R2-$i-400-$j-fc.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in 0; do cut -f4 dev-n/out/dev-R2-$i-400-$j-fc.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) > dev-n/data/minmaxavgGeometric.R2.dat
```

## Plotten die hap
```gnuplot
set term x11 1
set logscale y
set xlabel "gamma"
set ylabel "perplexity"
set title "R2"
set yrange [1:100000]

plot "minmaxavgGeometric.R2.dat" u 1:4 title "avgGeo" w l linecolor rgb "black", "minmaxavgLinear.R2.dat" u 1:7 title "avg lsalm" w l linecolor rgb "blue", "minmaxavgLinear.R2.dat" u 1:10 title "avg ngram" w l linecolor rgb "blue", "minmaxavgLinear.R2.dat" u 1:4 title "avgLin" w l linecolor rgb "blue"
```

set term x11 0
set logscale y
set xlabel "gamma"
set ylabel "perplexity"
set title "R3"
set yrange [1:100000]

plot "minmaxavgGeometric.R3.dat" u 1:4 title "avgGeo" w l linecolor rgb "black", "minmaxavgLinear.R3.dat" u 1:7 title "avg lsalm" w l linecolor rgb "blue", "minmaxavgLinear.R3.dat" u 1:10 title "avg ngram" w l linecolor rgb "blue", "minmaxavgLinear.R3.dat" u 1:4 title "avgLin" w l linecolor rgb "blue"




# Scratch pad

paste <(for i in `seq 0.00 0.10 0.90`; do echo $i; done) <(for i in `seq 0.00 0.10 0.90`; do ( for j in 0; do cut -f2 dev-n/out/dev-R2-$i-400-$j-fc.lin1.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}'; done) <(for i in `seq 0.00 0.10 0.90`; do ( for j in 0; do cut -f3 dev-n/out/dev-R2-$i-400-$j-fc.lin1.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) <(for i in `seq 0.00 0.10 0.90`; do ( for j in 0; do cut -f4 dev-n/out/dev-R2-$i-400-$j-fc.lin1.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) > dev-n/data/minmaxavgLinear.R2-1.dat

set term x11 2
set logscale y
set xlabel "gamma"
set ylabel "perplexity"
set title "R3"
set yrange [1:100000]

plot "minmaxavgLinear.R2-1.dat" u 1:4 title "avgGeo" w l linecolor rgb "black", "minmaxavgLinear.R3.dat" u 1:7 title "avg lsalm" w l linecolor rgb "blue", "minmaxavgLinear.R2-1.dat" u 1:10 title "avg ngram" w l linecolor rgb "red", "minmaxavgLinear.R2-1.dat" u 1:4 title "avgLin" w l linecolor rgb "magenta"


paste <(for i in `seq 0.00 0.05 1.50`; do echo $i; done) <(for i in `seq 0.00 0.05 1.50`; do ( for j in 0; do cut -f2 dev-n/out/dev-R2-$i-400-$j-fc.lin3.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}'; done) <(for i in `seq 0.00 0.05 1.50`; do ( for j in 0; do cut -f3 dev-n/out/dev-R2-$i-400-$j-fc.lin3.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) <(for i in `seq 0.00 0.05 1.50`; do ( for j in 0; do cut -f4 dev-n/out/dev-R2-$i-400-$j-fc.lin3.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) > dev-n/data/minmaxavgLinear.R2-3.dat

plot "minmaxavgLinear.R2-3.dat" u 1:4 title "avgGeo" w l linecolor rgb "black", "minmaxavgLinear.R2-3.dat" u 1:7 title "avg lsalm" w l linecolor rgb "blue", "minmaxavgLinear.R2-3.dat" u 1:10 title "avg ngram" w l linecolor rgb "red", "minmaxavgLinear.R2-3.dat" u 1:4 title "avgLin" w l linecolor rgb "magenta"





paste <(for i in `seq 0.00 0.05 2.50`; do echo $i; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in 0; do cut -f2 dev-n/out/dev-R2-$i-400-$j-fc.geo3.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}'; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in 0; do cut -f3 dev-n/out/dev-R2-$i-400-$j-fc.geo3.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) <(for i in `seq 0.00 0.05 2.50`; do ( for j in 0; do cut -f4 dev-n/out/dev-R2-$i-400-$j-fc.geo3.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) > dev-n/data/minmaxavgGeometric.R2-3.dat

plot "minmaxavgGeometric.R2-3.dat" u 1:4 title "avgGeo" w l linecolor rgb "black", "minmaxavgLinear.R2-3.dat" u 1:7 title "avg lsalm" w l linecolor rgb "blue", "minmaxavgLinear.R2-3.dat" u 1:10 title "avg ngram" w l linecolor rgb "red", "minmaxavgLinear.R2-3.dat" u 1:4 title "avgLin" w l linecolor rgb "magenta"






paste <(for i in `seq 0.00 1.00 11.00`; do echo $i; done) <(for i in `seq 0.00 1.00 11.00`; do ( for j in 0; do cut -f2 dev-n/out/dev-R2-$i-400-$j-fc.lin3.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}'; done) <(for i in `seq 0.00 1.00 11.00`; do ( for j in 0; do cut -f3 dev-n/out/dev-R2-$i-400-$j-fc.lin3.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) <(for i in `seq 0.00 1.00 11.00`; do ( for j in 0; do cut -f4 dev-n/out/dev-R2-$i-400-$j-fc.lin3.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) > dev-n/data/minmaxavgLinear.R2-31.dat

paste <(for i in `seq 0.00 1.00 11.00`; do echo $i; done) <(for i in `seq 0.00 1.00 11.00`; do ( for j in 0; do cut -f2 dev-n/out/dev-R2-$i-400-$j-fc.geo3.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}'; done) <(for i in `seq 0.00 1.00 11.00`; do ( for j in 0; do cut -f3 dev-n/out/dev-R2-$i-400-$j-fc.geo3.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) <(for i in `seq 0.00 1.00 11.00`; do ( for j in 0; do cut -f4 dev-n/out/dev-R2-$i-400-$j-fc.geo3.out | ppl | cut -f2 -d' '; done ) | awk 'NR == 1 { max=$1; min=$1; sum=0 } { if ($1>max) max=$1; if ($1<min) min=$1; sum+=$1;} END {printf "%f\t%f\t%f\n", min, max, sum/NR}' ; done) > dev-n/data/minmaxavgGeometric.R2-31.dat
