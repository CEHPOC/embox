/**
 * \file net_pack.h
 *
 * \date Mar 7, 2009
 * \author anton
 */

#ifndef NET_PACK_H_
#define NET_PACK_H_

net_packet *net_packet_alloc();
void net_packet_free(net_packet *pack);
net_packet *net_packet_copy(net_packet *pack);
/*
int net_packet_manager_init();
void *net_pack_manager_alloc();

void *net_pack_manager_lock_pack(void *manager, const unsigned char * data, unsigned short len);
void net_pack_manager_unlock_pack(void *manager, void *net_pack_info);
*/
#endif /* NET_PACK_H_ */
