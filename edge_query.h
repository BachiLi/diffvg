#pragma once

struct EdgeQuery {
	int shape_group_id;
    int shape_id;
    bool hit; // Do we hit the specified shape_group_id & shape_id?
};
