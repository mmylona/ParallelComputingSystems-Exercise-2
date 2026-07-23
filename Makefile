CC      = gcc
CFLAGS  = -O2 -Wall
LDFLAGS = -lpthread

EXERCISES = Exercise1 Exercise2 Exercise3  

.PHONY: all run clean $(EXERCISES)

all: $(EXERCISES)

$(EXERCISES):
	$(MAKE) -s -C $@

run: all
	@for ex in $(EXERCISES); do \
		script=$$(ls $$ex/*.sh 2>/dev/null | head -1); \
		if [ -n "$$script" ]; then \
			echo ""; \
			echo "========================================"; \
			bash $$script; \
		fi \
	done

clean:
	@for ex in $(EXERCISES); do \
		$(MAKE) -s -C $$ex clean; \
	done