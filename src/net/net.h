/**
 * \file net.h
 *
 * \date Mar 11, 2009
 * \author anton
 */

#ifndef NET_H_
#define NET_H_

#include "net_device.h"
#include "eth.h"

#include "sock.h"


int netif_rx(net_packet *pack);

#endif /* NET_H_ */
