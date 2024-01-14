
fps=10
duration=10
frames=$((fps * duration))
mkdir -p frames
echo >Makefile

allframes=

for i in `seq $frames`; do
    fn=$(printf frames/%03d.png $i)
    allframes="$allframes $fn"
    t=$(echo "scale=10; ($i-1)/$frames" | bc)
    cat - <<END >>Makefile
$fn:
	echo "\$@"
	@python julia.py -r 360 240 -s 6 -t 8 -c -o \$@ -a $t

END
done

cat - <<END >>Makefile

all: $allframes ;

frames/julia_anim.mp4: $allframes
	ffmpeg -framerate $fps -pattern_type glob -i 'frames/*.png' -c:v libx264 -pix_fmt yuv420p -crf 24 \$@

END

