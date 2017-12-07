#!/bin/bash
# credits: https://github.com/TheDuck314/go-NN


mkdir -p /tmp/data

cd /tmp/data

# ------------------------

wget --output-document=- 'http://www.u-go.net/gamerecords-4d' | grep 'Download' | sed 's/.*"\(.*\)".*/\1/' > links.txt
while read link ; do
   wget "$link"    
done < links.txt

for archive in *.tar.gz ; do
    echo "extract $archive"
    tar xzf "$archive"
done

# ------------------------

wget --output-document=- http://u-go.net/gamerecords/ | grep Download | grep bz2 | sed 's/.*"\(.*\)".*/\1/' > links.txt
while read link ; do
    echo "fetching $link"
    wget "$link"    
done < links.txt

for archive in *.tar.gz ; do
    echo "extract " $archive
    tar xzf "$archive"
done