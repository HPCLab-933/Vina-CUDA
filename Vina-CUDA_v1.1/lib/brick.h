/*

   Copyright (c) 2006-2010, The Scripps Research Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Author: Dr. Oleg Trott <ot14@columbia.edu>, 
           The Olson Lab, 
           The Scripps Research Institute

*/

#ifndef VINA_BRICK_H
#define VINA_BRICK_H

#include "common.h"


inline fl closest_between(fl begin, fl end, fl x) {
	assert(begin <= end);       
	if(x <= begin) return begin;
	else if(x >= end) return end;
	return x;
}


inline vec brick_closest(const vec& begin, const vec& end, const vec& v) {
	vec tmp;
	VINA_FOR_IN(i, tmp)             // for (i = 0; i<tmp.size; i++)
		tmp[i] = closest_between(begin[i], end[i], v[i]);
	return tmp;
}



inline fl brick_distance_sqr(const vec& begin, const vec& end, const vec& v) {
	vec closest; closest = brick_closest(begin, end, v);
	return vec_distance_sqr(closest, v);
}

#endif
