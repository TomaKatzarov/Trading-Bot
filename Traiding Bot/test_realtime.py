async def live_test(symbol='AAPL', test_duration=60):
    exiter = GracefulExiter()
    handler = None
    
    try:
        handler = RealTimeDataHandler(symbol=symbol)
        logger.info(f"Starting {symbol} stream...")
        
        start_time = datetime.now()
        await handler.start()
        
        while not exiter.should_exit:
            await asyncio.sleep(1)
            
            recent = handler.get_recent_data(1)
            if recent:
                logger.info(f"Latest: {recent[0]['timestamp']} | Close: ${recent[0]['close']:.2f}")
            
            if (datetime.now() - start_time).total_seconds() > test_duration:
                break
                
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False
    finally:
        if handler:
            await handler.stop()
        return True