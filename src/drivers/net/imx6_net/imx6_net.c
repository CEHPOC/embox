/**
 * @file imx6_net.c
 * @brief iMX6 MAC-NET driver
 * @author Denis Deryugin <deryugin.denis@gmail.com>
 * @version 0.1
 * @date 2016-05-11
 */

#include <assert.h>
#include <errno.h>
#include <string.h>

#include <hal/reg.h>

#include <kernel/irq.h>
#include <kernel/printk.h>

#include <net/inetdevice.h>
#include <net/l0/net_entry.h>
#include <net/l2/ethernet.h>
#include <net/netdevice.h>
#include <net/skbuff.h>

#include <util/log.h>

#include <embox/unit.h>

#include <framework/mod/options.h>

#include "imx6_net.h"

static void _reg_dump(void) {
	log_debug("ENET_EIR  %10x", REG32_LOAD(ENET_EIR ));
	log_debug("ENET_EIMR %10x", REG32_LOAD(ENET_EIMR));
	log_debug("ENET_RDAR %10x", REG32_LOAD(ENET_RDAR));
	log_debug("ENET_TDAR %10x", REG32_LOAD(ENET_TDAR));
	log_debug("ENET_ECR  %10x", REG32_LOAD(ENET_ECR ));
	log_debug("ENET_MSCR %10x", REG32_LOAD(ENET_MSCR));
	log_debug("ENET_RCR  %10x", REG32_LOAD(ENET_RCR ));
	log_debug("ENET_TCR  %10x", REG32_LOAD(ENET_TCR ));
	log_debug("MAC_LOW   %10x", REG32_LOAD(MAC_LOW  ));
	log_debug("MAC_HI    %10x", REG32_LOAD(MAC_HI   ));
	log_debug("ENET_IAUR %10x", REG32_LOAD(ENET_IAUR));
	log_debug("ENET_IALR %10x", REG32_LOAD(ENET_IALR));
	log_debug("ENET_GAUR %10x", REG32_LOAD(ENET_GAUR));
	log_debug("ENET_GALR %10x", REG32_LOAD(ENET_GALR));
	log_debug("ENET_TFWR %10x", REG32_LOAD(ENET_TFWR));
	log_debug("ENET_RDSR %10x", REG32_LOAD(ENET_RDSR));
	log_debug("ENET_TDSR %10x", REG32_LOAD(ENET_TDSR));
	log_debug("ENET_MRBR %10x", REG32_LOAD(ENET_MRBR));
}

static uint32_t _uboot_regs[0x200];
static void _mem_dump(void) {
	_uboot_regs[ENET_EIR  >> 2] = REG32_LOAD(ENET_EIR );
	_uboot_regs[ENET_EIMR >> 2] = REG32_LOAD(ENET_EIMR);
	_uboot_regs[ENET_RDAR >> 2] = REG32_LOAD(ENET_RDAR);
	_uboot_regs[ENET_TDAR >> 2] = REG32_LOAD(ENET_TDAR);
	_uboot_regs[ENET_ECR  >> 2] = REG32_LOAD(ENET_ECR );
	_uboot_regs[ENET_MSCR >> 2] = REG32_LOAD(ENET_MSCR);
	_uboot_regs[ENET_RCR  >> 2] = REG32_LOAD(ENET_RCR );
	_uboot_regs[ENET_TCR  >> 2] = REG32_LOAD(ENET_TCR );
	_uboot_regs[MAC_LOW   >> 2] = REG32_LOAD(MAC_LOW  );
	_uboot_regs[MAC_HI    >> 2] = REG32_LOAD(MAC_HI   );
	_uboot_regs[ENET_IAUR >> 2] = REG32_LOAD(ENET_IAUR);
	_uboot_regs[ENET_IALR >> 2] = REG32_LOAD(ENET_IALR);
	_uboot_regs[ENET_GAUR >> 2] = REG32_LOAD(ENET_GAUR);
	_uboot_regs[ENET_GALR >> 2] = REG32_LOAD(ENET_GALR);
	_uboot_regs[ENET_TFWR >> 2] = REG32_LOAD(ENET_TFWR);
	_uboot_regs[ENET_RDSR >> 2] = REG32_LOAD(ENET_RDSR);
	_uboot_regs[ENET_TDSR >> 2] = REG32_LOAD(ENET_TDSR);
	_uboot_regs[ENET_MRBR >> 2] = REG32_LOAD(ENET_MRBR);
}

static void _mem_restore(void) {
	REG32_LOAD(ENET_EIR ) = _uboot_regs[ENET_EIR  >> 2];
	REG32_LOAD(ENET_EIMR) = _uboot_regs[ENET_EIMR >> 2];
	REG32_LOAD(ENET_RDAR) = _uboot_regs[ENET_RDAR >> 2];
	REG32_LOAD(ENET_TDAR) = _uboot_regs[ENET_TDAR >> 2];
	REG32_LOAD(ENET_ECR ) = _uboot_regs[ENET_ECR  >> 2];
	REG32_LOAD(ENET_MSCR) = _uboot_regs[ENET_MSCR >> 2];
	REG32_LOAD(ENET_RCR ) = _uboot_regs[ENET_RCR  >> 2];
	REG32_LOAD(ENET_TCR ) = _uboot_regs[ENET_TCR  >> 2];
	REG32_LOAD(MAC_LOW  ) = _uboot_regs[MAC_LOW   >> 2];
	REG32_LOAD(MAC_HI   ) = _uboot_regs[MAC_HI    >> 2];
	REG32_LOAD(ENET_IAUR) = _uboot_regs[ENET_IAUR >> 2];
	REG32_LOAD(ENET_IALR) = _uboot_regs[ENET_IALR >> 2];
	REG32_LOAD(ENET_GAUR) = _uboot_regs[ENET_GAUR >> 2];
	REG32_LOAD(ENET_GALR) = _uboot_regs[ENET_GALR >> 2];
	REG32_LOAD(ENET_TFWR) = _uboot_regs[ENET_TFWR >> 2];
	REG32_LOAD(ENET_RDSR) = _uboot_regs[ENET_RDSR >> 2];
	REG32_LOAD(ENET_TDSR) = _uboot_regs[ENET_TDSR >> 2];
	REG32_LOAD(ENET_MRBR) = _uboot_regs[ENET_MRBR >> 2];
}

static void emac_set_macaddr(unsigned char (*_macaddr)[6]) {
	uint32_t mac_hi, mac_lo;

	mac_hi  = (*_macaddr[5] << 16) |
	          (*_macaddr[4] << 24);
	mac_lo  = (*_macaddr[3] <<  0) |
	          (*_macaddr[2] <<  8) |
	          (*_macaddr[1] << 16) |
	          (*_macaddr[0] << 24);

	REG32_STORE(MAC_LOW, mac_lo);
	REG32_STORE(MAC_HI, mac_hi);
}

static struct imx6_buf_desc _tx_desc_ring[TX_BUF_FRAMES] __attribute__ ((aligned(0x10)));
static struct imx6_buf_desc _rx_desc_ring[RX_BUF_FRAMES] __attribute__ ((aligned(0x10)));

static int _cur_rx = 0;
static int _dirty_tx = 0;
static int _cur_tx = 0;

static uint8_t _tx_buf[TX_BUF_FRAMES][2048] __attribute__ ((aligned(0x10)));
static uint8_t _rx_buf[RX_BUF_FRAMES][2048] __attribute__ ((aligned(0x10)));

extern void dcache_inval(const void *p, size_t size);
extern void dcache_flush(const void *p, size_t size);

static int imx6_net_xmit(struct net_device *dev, struct sk_buff *skb) {
	uint8_t *data;
	struct imx6_buf_desc *desc;

	log_debug("Transmitting packet %2d", _cur_tx);

	ipl_t sp;

	assert(dev);
	assert(skb);

	sp = ipl_save();
	{
		REG32_STORE(ENET_TCR, (1 << 2));
		data = (uint8_t*) skb_data_cast_in(skb->data);

		if (!data) {
			log_error("No skb data!\n");
			ipl_restore(sp);
			return -1;
		}
		memset(&_tx_buf[_cur_tx][0], 0, 2048);
		memcpy(&_tx_buf[_cur_tx][0], data, skb->len);
		dcache_flush(&_tx_buf[_cur_tx][0], skb->len);

		desc = &_tx_desc_ring[_cur_tx];
		desc->data_pointer = (uint32_t) &_tx_buf[_cur_tx][0];
		desc->len          = skb->len;
		desc->flags1       = FLAG_L | FLAG_TC;

		skb_free(skb);

		if (_cur_tx == TX_BUF_FRAMES - 1)
			desc->flags1 |= FLAG_W;

		desc->flags1 |= FLAG_R;

		dcache_flush(desc, sizeof(struct imx6_buf_desc));

		REG32_STORE(ENET_TDAR, 0xFFFFFFFF);

		int timeout = 0xFF;
		while(timeout--) {
			if (!(REG32_LOAD(ENET_TDAR)))
				break;
		}

		if (timeout == 0)
			log_debug("TX timeout...");

		//REG32_STORE(ENET_TCR, (1 << 2) | 1);
	}
	ipl_restore(sp);

	_cur_tx = (_cur_tx + 1) % TX_BUF_FRAMES;

	return 0;
}

static void _init_buffers(void) {
	struct imx6_buf_desc *desc;

	memset(&_tx_desc_ring[0], 0,
	        TX_BUF_FRAMES * sizeof(struct imx6_buf_desc));
	memset(&_rx_desc_ring[0], 0,
	        RX_BUF_FRAMES * sizeof(struct imx6_buf_desc));

	/* Mark last buffer (i.e. set wrap flag) */
	_rx_desc_ring[RX_BUF_FRAMES - 1].flags1 = FLAG_W;
	_tx_desc_ring[TX_BUF_FRAMES - 1].flags1 = FLAG_W;

	for (int i = 0; i < RX_BUF_FRAMES; i++) {
		desc = &_rx_desc_ring[i];
		desc->data_pointer = (uint32_t) &_rx_buf[i][0];
		desc->flags1 |= FLAG_E;
	}

	dcache_flush(&_tx_desc_ring[0],
	              TX_BUF_FRAMES * sizeof(struct imx6_buf_desc));
	dcache_flush(&_rx_desc_ring[0],
	              RX_BUF_FRAMES * sizeof(struct imx6_buf_desc));

	assert((((uint32_t) &_rx_desc_ring[0]) & 0xF) == 0);
	REG32_STORE(ENET_RDSR, ((uint32_t) &_rx_desc_ring[0]));

	assert((((uint32_t) &_tx_desc_ring[0]) & 0xF) == 0);
	REG32_STORE(ENET_TDSR, ((uint32_t) &_tx_desc_ring[0]));

	assert((RX_BUF_LEN & 0xF) == 0);
	REG32_STORE(ENET_MRBR, (FRAME_LEN - 1));
}

#if 0
static void _reg_setup(void) {
	uint32_t t;

	/* Disable IRQ */
	REG32_STORE(ENET_EIMR, 0);

	/* Clear pending interrupts */
	//REG32_STORE(ENET_EIR, EIR_MASK);

	/* Setup RX */
	t  = (FRAME_LEN - 1) << FRAME_LEN_OFFSET;
	t |= RCR_FCE | RCR_MII_MODE;
	REG32_STORE(ENET_RCR, t);

	_setup_mii_speed();

	/* Enable IRQ */
	REG32_STORE(ENET_EIMR, 0x7FFF0000);
}
#endif

static void _reset(void) {
	int cnt = 0;

	log_debug("ENET reset...\n");
	_mem_dump();

	REG32_STORE(ENET_ECR, RESET);
	while(REG32_LOAD(ENET_ECR) & RESET){
		if (cnt ++ > 100000) {
			log_error("enet can't be reset");
			break;
		}
	}
	//REG32_STORE(ENET_ECR, 0xF0000100);
	_mem_restore();

	_init_buffers();

	REG32_STORE(ENET_EIMR, 0xFFFFFFFF);
	REG32_STORE(ENET_EIR, 0xFFFFFFFF);
	//REG32_STORE(ENET_EIMR, 0x7FFF0000);
	//REG32_STORE(ENET_EIR, 0x7FFF0000);
	REG32_STORE(ENET_TCR, (1 << 2));

	//REG32_STORE(ENET_TFWR, 0x0);

	uint32_t t = 0x08000124;
	REG32_STORE(ENET_RCR, t);

	t = REG32_LOAD(ENET_ECR);
	t |= ETHEREN;
	REG32_STORE(ENET_ECR, t); /* Note: should be last ENET-related init step */
	_reg_dump();

	REG32_STORE(ENET_RDAR, (1 << 24));
/*
	while(REG32_LOAD(ENET_RDAR)) {
		log_debug("ENET_RDAR not zero");
		for (int id = 0; id < RX_BUF_FRAMES; id++) {
			dcache_inval(&_rx_desc_ring[id], sizeof(struct imx6_buf_desc));
			if (!(_rx_desc_ring[id].flags1 & FLAG_E))
				log_debug("Frame %2d status %#06x", id, _rx_desc_ring[id].flags1);
		}
	}
*/
	//uint32_t t = REG32_LOAD(ENET_RCR);
	//t |= 1;
	//REG32_STORE(ENET_RCR, t);


#if 0
	int tmp = 10000;


	REG32_STORE(ENET_ECR, RESET);
	while(tmp--);

	_write_macaddr();

	//REG32_STORE(ENET_EIR, 0xFFC00000);

	REG32_STORE(ENET_IAUR, 0);
	REG32_STORE(ENET_IALR, 0);
	REG32_STORE(ENET_GAUR, 0);
	REG32_STORE(ENET_GALR, 0);

	assert((RX_BUF_LEN & 0xF) == 0);
	REG32_STORE(ENET_MRBR, RX_BUF_LEN);

	assert((((uint32_t) &_rx_desc_ring[0]) & 0xF) == 0);
	REG32_STORE(ENET_RDSR, ((uint32_t) &_rx_desc_ring[0]));

	assert((((uint32_t) &_tx_desc_ring[0]) & 0xF) == 0);
	REG32_STORE(ENET_TDSR, ((uint32_t) &_tx_desc_ring[0]));

	/* dirty_tx, cur_tx, cur_rx */

	for (int i = 0; i < TX_BUF_FRAMES; i++)
		memset(&_tx_desc_ring[i], 0, sizeof(_tx_desc_ring[i]));

	REG32_STORE(ENET_TCR, (1 << 2) | 1);

	//REG32_STORE(ENET_MSCR, 0x8); /* TODO fix? */
	_setup_mii_speed();

	uint32_t rcntl = 0x40000000 | 0x00000020;
	rcntl &= ~(1 << 8);
	/* 10 mbps */
	rcntl |= (1 << 9);
	REG32_STORE(ENET_RCR, rcntl);

	uint32_t ecntl = 0x2 | (1 << 8);
	REG32_STORE(ENET_TFWR, (1 << 8));

	REG32_STORE(ENET_ECR, ecntl);

	REG32_STORE(ENET_RDAR, 0);

	REG32_STORE(ENET_EIMR, 0x0A800000);

	return;

	_reg_dump();
	_reg_setup();


	/* TODO set FEC clock? */


	_init_buffers();

	REG32_STORE(ENET_EIMR, EIMR_TXF | EIMR_RXF | EIR_MASK);

	_write_macaddr();

	REG32_STORE(ENET_TFWR, (1 << 8) | 0x1);
	REG32_STORE(ENET_ECR, ETHEREN | ECR_DBSWP); /* Note: should be last init step */
#endif
}

static int imx6_net_open(struct net_device *dev) {
	return 0;
}

static int imx6_net_set_macaddr(struct net_device *dev, const void *addr) {
	assert(dev);
	assert(addr);

	emac_set_macaddr((unsigned char (*)[6])addr);

	return 0;
}

static irq_return_t imx6_irq_handler(unsigned int irq_num, void *dev_id) {
	uint32_t state;
	struct imx6_buf_desc *desc;

	state = REG32_LOAD(ENET_EIR);

	log_debug("Interrupt mask %#010x", state);

	REG32_STORE(ENET_EIR, state);

	REG32_STORE(ENET_RDAR, 1 << 24);

	if (state == 0x10000000)
		return IRQ_HANDLED;

	if (state & EIR_EBERR) {
		log_error("Ethernet bus error, resetting ENET!");
		REG32_STORE(ENET_ECR, RESET);
		_reset();

		return IRQ_HANDLED;
	}

	if (state & (EIR_RXB | EIR_RXF)) {
		log_debug("RX interrupt");
		desc = &_rx_desc_ring[_cur_rx];
		dcache_inval(desc, sizeof(struct imx6_buf_desc));
		dcache_inval((void *)desc->data_pointer, 2048);

		if (desc->flags1 & FLAG_E) {
			log_error("Current RX descriptor is empty!");
		} else {
			struct sk_buff *skb = skb_alloc(desc->len);
			assert(skb);
			skb->len = desc->len;
			skb->dev = dev_id;
			memcpy(skb_data_cast_in(skb->data),
				(void*)desc->data_pointer, desc->len);
			netif_rx(skb);

			desc->flags1 = FLAG_E;
			if (_cur_rx == RX_BUF_FRAMES - 1)
				desc->flags1 |= FLAG_W;
			dcache_flush(desc, sizeof(struct imx6_buf_desc));
			_cur_rx = (_cur_rx + 1) % RX_BUF_FRAMES;
		}
	}

	if (state & (EIR_TXB | EIR_TXF)) {
		log_debug("finished TX");
		//_flag = 0;
		desc = &_tx_desc_ring[_dirty_tx];
		dcache_inval(desc, sizeof(struct imx6_buf_desc));
		if (desc->flags1 & FLAG_R)
			log_error("No single frame transmitted!");

		while (!(desc->flags1 & FLAG_R)) {
			assert(desc->data_pointer == (uint32_t) &_tx_buf[_dirty_tx]);
			log_debug("Frame %2d transmitted", _dirty_tx);
			_dirty_tx = (_dirty_tx + 1) % TX_BUF_FRAMES;
			desc = &_tx_desc_ring[_dirty_tx];
			dcache_inval(desc, sizeof(struct imx6_buf_desc));
		}
	}

	REG32_STORE(ENET_TCR, 0x4);

	_reg_dump();

	return IRQ_HANDLED;
}

static const struct net_driver imx6_net_drv_ops = {
	.xmit = imx6_net_xmit,
	.start = imx6_net_open,
	.set_macaddr = imx6_net_set_macaddr
};

EMBOX_UNIT_INIT(imx6_net_init);
static int imx6_net_init(void) {
	struct net_device *nic;
	int tmp;

	if (NULL == (nic = etherdev_alloc(0))) {
                return -ENOMEM;
        }

	nic->drv_ops = &imx6_net_drv_ops;

	tmp = irq_attach(ENET_IRQ, imx6_irq_handler, 0, nic, "i.MX6 enet");
	if (tmp)
		return tmp;

	_reset();

	return inetdev_register_dev(nic);
}
