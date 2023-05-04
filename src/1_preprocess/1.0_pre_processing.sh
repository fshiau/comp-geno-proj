#!/bin/bash

##Change path depending on raw-data directory
DATADIR=/gpfs/commons/home/cangel/g2lab/courses/ComputationalGenomics/project/data/ATAC-seq/signal
DIR=$(dirname ${DATADIR})
mkdir -p $DIR/pbs_prom/

### For RNA-seq analysis we downloaded the authors data and obtained the reads by using salmon

for SRR in $(find $DATADIR -name "*.bedgraph" -type f); do
FILENAME=${SRR##*/}    # Extract the file name
FILENAME2=${FILENAME%.bigWig.bedgraph}
DIRPATH=${SRR%/*}      # Extract the directory path
echo "File name: $FILENAME"
echo "Directory path: $DIRPATH"
cat <<EOT >> $DIR/pbs_prom/${FILENAME2}.sh
#!/bin/bash
#SBATCH --job-name=${FILENAME2}.%J.job
#SBATCH --output=$DIR/pbs_prom/${FILENAME2}.%J.out
#SBATCH --error=$DIR/pbs_prom/${FILENAME2}.%J.error
#SBATCH --time=01:05:00
#SBATCH --mem=10G

#USAGE:

module load bedtools/2.27.1
module load kentutils/302.1

SIZE=/gpfs/commons/home/cangel/courses/ComputationalBiology/project/old_data/resources/hg19.chrom.sizes

mkdir -p ${DIRPATH}/processed_prom

cd ${DIRPATH}

echo "Sample ${SRR}"
egrep "chr([1-9]|1[0-9]|2[0-2])" ${DIRPATH}/$FILENAME > ${DIRPATH}/processed_prom/$FILENAME2.sorted
bedtools map -a /gpfs/commons/home/cangel/courses/ComputationalBiology/project/scripts/hg19_promoters_noName.tsv  -b ${DIRPATH}/processed_prom/$FILENAME2.sorted -c 4 -o mean -sorted > ${DIRPATH}/processed_prom/$FILENAME2.promoters
rm ${DIRPATH}/processed_prom/$FILENAME2.sorted

EOT
sbatch $DIR/pbs_prom/${FILENAME2}.sh

done

