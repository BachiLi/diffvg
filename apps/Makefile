SEAM_IMAGES=seaside2 sunset2 hokusai cat ice_cream
SEAM_OUT=results/seam_carving
SEAM_RESULTS=$(addsuffix /out.mp4,$(addprefix $(SEAM_OUT)/,$(SEAM_IMAGES)))

all:  $(SEAM_RESULTS)
	echo $(SEAM_RESULTS)

$(SEAM_OUT)/%/out.mp4: imgs/seamcarving/%.svg
	python seam_carving.py --svg $^

