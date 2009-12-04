/**
 * @file ip.c
 *
 * @brief The Internet Protocol (IP) module.
 * @date 17.03.2009
 * @author Alexandr Batyukov
 * @author Nikolay Korotky
 */
#include "err.h"
#include "net/net.h"
#include "net/skbuff.h"
#include "lib/inet/netinet/in.h"
#include "net/ip.h"
#include "net/inet_sock.h"
#include "net/if_ether.h"
#include "net/netdevice.h"
#include "net/inetdevice.h"
#include "net/route.h"
#include "net/checksum.h"

int ip_rcv(sk_buff_t *pack) {
	LOG_DEBUG("ip packet received\n");
	pack->h.raw = pack->nh.raw + IP_HEADER_SIZE;
	net_device_stats_t *stats = pack->netdev->netdev_ops->ndo_get_stats(pack->netdev);
	iphdr_t *iph = pack->nh.iph;
	/**
	 *   RFC1122: 3.1.2.2 MUST silently discard any IP frame that fails the checksum.
	 *   Is the datagram acceptable?
	 *   1.  Length at least the size of an ip header
	 *   2.  Version of 4
	 *   3.  Checksums correctly. [Speed optimisation for later, skip loopback checksums]
	 *   4.  Doesn't have a bogus length
	 */
	if (iph->ihl < 5 || iph->version != 4) {
		LOG_ERROR("invalide IPv4 header\n");
		stats->rx_err += 1;
		return -1;
	}
	unsigned short tmp = iph->check;
	iph->check = 0;
	if (tmp != ptclbsum(pack->nh.raw, IP_HEADER_SIZE)) {
		LOG_ERROR("bad ip checksum\n");
		stats->rx_crc_errors += 1;
		return -1;
	}

	unsigned int len = ntohs(iph->tot_len);
	if (pack->len < len || len < (iph->ihl * 4)) {
		LOG_ERROR("invalide IPv4 header length\n");
		stats->rx_length_errors += 1;
		return -1;
	}
	/**
	 * Check the destination address, and if it dosn't match
	 * any of own addresses, retransmit packet according to routing table.
	 */
	if(ip_dev_find(pack->nh.iph->daddr) == NULL) {
		if(!ip_route(pack)) {
			dev_queue_xmit(pack);
		}
	        return 0;
	}
	if (ICMP_PROTO_TYPE == iph->proto) {
		icmp_rcv(pack);
	}
	if (UDP_PROTO_TYPE == iph->proto) {
		udp_rcv(pack);
	}
	return 0;
}
