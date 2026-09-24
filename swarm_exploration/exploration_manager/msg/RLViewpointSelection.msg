std_msgs/Header header
uint32 task_id
int32 drone_id
int32 candidate_index
# x/y are offsets in the UAV body frame at selection time; z is vertical.
geometry_msgs/Vector3 position_offset
float64 yaw_offset
