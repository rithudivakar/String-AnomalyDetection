from controller.controller import AnomalyDetectionController

def main():
    k = 5
    controller = AnomalyDetectionController(k)
    controller.load_data("/home/rithuparnakd/projectData/AnomalyDetection/anomalyDetectionString/data/datadata.csv")
    anomalies = controller.detect_anomalies()
    print(f"Found {len(anomalies)} anomalies:")
    for anomaly in anomalies:
        print(f"  {anomaly.record}")
        
if __name__ == "__main__":
    main()
