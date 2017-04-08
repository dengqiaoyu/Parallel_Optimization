/**
 * @file return_error.h
 * @brief This file contains the return error define for master.c and worker.c
 * @author Qiaoyu Deng(qdeng), Changkai Zhou(zchangka)
 * @bug No known bugs
 */
#ifndef _RETURN_ERROR_H_
#define _RETURN_ERROR_H_

#define SUCCESS 0

#define RETURN_IF_ERROR(f, error_type) if((f) != SUCCESS) return (error_type)

/* worker_sche_queue.c */
#define SHCE_QUEUE_ERROR_INIT_SCHE_LOCK_FAILED -1
#define SHCE_QUEUE_ERROR_INIT_SCHE_COND_FAILED -2
#define FIFO_QUEUE_ERROR_INIT_FIFO_LOCK_FAILED -3
#define FAST_QUEUE_ERROR_INIT_LOCK_FAILED -4

#endif /* _RETURN_ERROR_H_ */