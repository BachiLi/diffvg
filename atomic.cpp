//A hacky solution to get around the Ellipse include

#ifdef WIN32
#include <windows.h>
#include <cstdint>

float win_atomic_add(float &target, float source) {
	union { int i; float f; } old_val;
	union { int i; float f; } new_val;
	do {
		old_val.f = target;
		new_val.f = old_val.f + (float)source;
	} while (InterlockedCompareExchange((LONG*)&target, (LONG)new_val.i, (LONG)old_val.i) != old_val.i);
	return old_val.f;
}

double win_atomic_add(double &target, double source) {
	union { int64_t i; double f; } old_val;
	union { int64_t i; double f; } new_val;
	do {
		old_val.f = target;
		new_val.f = old_val.f + (double)source;
	} while (InterlockedCompareExchange64((LONG64*)&target, (LONG64)new_val.i, (LONG64)old_val.i) != old_val.i);
	return old_val.f;
}

#endif