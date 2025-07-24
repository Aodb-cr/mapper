def example_usage():
    """ตัวอย่างการใช้งาน learning system"""
    
    from universal_field_extractor import SmartFieldExtractor
    
    # สร้าง extractor (จะมี learning อัตโนมัติ)
    extractor = SmartFieldExtractor()
    
    # ข้อมูลตัวอย่าง
    sample_data = {
        "alert_info": {
            "alert_title": "Suspicious Login Detected",
            "risk_level": "High",
            "event_timestamp": "2024-01-15T10:30:00Z"
        },
        "source_host": {
            "computer_name": "SERVER-01",
            "ip_addr": "10.0.1.50"
        },
        "account_info": {
            "username": "admin@company.com"
        }
    }
    
    print("🔍 Processing with learning...")
    
    # Extract และเรียนรู้อัตโนมัติ
    results = extractor.extract_all_fields(sample_data, "security_event")
    
    print(f"\n📊 Results:")
    for field_name, result in results.items():
        if result.value:
            confidence = getattr(result, 'confidence', 0)
            method = getattr(result, 'method', 'unknown')
            print(f"   {field_name}: {result.value}")
            print(f"      └─ Source: {result.source_path} ({method}, {confidence:.2f})")
    
    # แสดงสถิติการเรียนรู้
    print(f"\n🧠 Learning Progress:")
    extractor.show_learning_stats()
    
    # Process ไฟล์ที่ 2 (จะใช้ความรู้จากไฟล์แรก)
    sample_data_2 = {
        "alert_info": {
            "alert_title": "Malware Execution",
            "risk_level": "Critical"
        },
        "source_host": {
            "computer_name": "WORKSTATION-05"
        }
    }
    
    print(f"\n🔍 Processing second file (with learned knowledge)...")
    results_2 = extractor.extract_all_fields(sample_data_2, "security_event")
    
    # แสดงว่า learning ช่วยอะไรได้บ้าง
    for field_name, result in results_2.items():
        if hasattr(result, 'method') and 'learning' in result.method:
            print(f"🧠 Learning helped: {field_name} = {result.value}")

if __name__ == "__main__":
    example_usage()