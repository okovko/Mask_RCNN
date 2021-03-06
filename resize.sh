function resize ()
{	while read line1; do
		magick convert $line1 -resize 256x256 "thumbnails/thumbnail_${line1}"
	done
}
ls *.jpg . | resize
