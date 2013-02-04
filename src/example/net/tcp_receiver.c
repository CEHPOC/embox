/**
 * @file
 * @brief
 *
 * @date 30.10.11
 * @author Alexander Kalmuk
 *	- UDP version
 * @author Anton Kozlov
 * @author Ilia Vaprol
 *	- TCP version
 */

#include <string.h>
#include <arpa/inet.h>
#include <fcntl.h>

#include <net/ip.h>
#include <net/socket.h>
#include <framework/example/self.h>
#include <prom/prom_printf.h>

EMBOX_EXAMPLE(exec);

#define LISTENING_PORT 20

static int exec(int argc, char **argv) {
	int sock, client, res;
	struct sockaddr_in addr;
	struct sockaddr_in dst;
	int dst_addr_len = 0;
	char buff[1024];
	int bytes_read;

	prom_printf("Hello. I'm tcp_receiver at %d port. I will print all received data.\n",
			LISTENING_PORT);

	res = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (res < 0) {
		prom_printf("can't create socket %d!\n", res);
		return res;
	}
	sock = res;

	addr.sin_family = AF_INET;
	addr.sin_port= htons(LISTENING_PORT);
	addr.sin_addr.s_addr = htonl(INADDR_ANY);

	res = bind(sock, (struct sockaddr *)&addr, sizeof(addr));
	if (res < 0) {
		prom_printf("error at bind() %d!\n", res);
		return res;
	}

	res = listen(sock, 1);
	if (res < 0) {
		prom_printf("error at listen() %d!\n", res);
		return res;
	}

	res = accept(sock, (struct sockaddr *)&dst, &dst_addr_len);
	if (res < 0) {
		prom_printf("error at accept() %d\n", res);
		return res;
	}
	client = res;

	prom_printf("client from %s:%d was connected\n",
			inet_ntoa(dst.sin_addr), ntohs(dst.sin_port));

	/* read from sock, print */
	while (1) {
		bytes_read = recvfrom(client, buff, sizeof buff, 0, NULL, NULL);
		if (bytes_read < 0) {
			break;
		}
		buff[bytes_read] = '\0';
		prom_printf("recv: '%s'\n", buff);
		if (strncmp(buff, "quit", 4) == 0) {
			prom_printf("client gonna to close connection\n");
			break;
		}
	}

	close(client);
	close(sock);

	prom_printf("Bye bye!\n");

	return 0;
}
