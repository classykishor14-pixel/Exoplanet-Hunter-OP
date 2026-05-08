import React, { useRef, useMemo, useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import { Canvas, useFrame } from '@react-three/fiber/native';
import * as THREE from 'three';
import { Gyroscope } from 'expo-sensors';
import { Colors } from '../theme';

function Stars() {
  const pointsRef = useRef<THREE.Points>(null);
  const gyroRef = useRef({ x: 0, y: 0, z: 0 });
  const currentRotation = useRef({ x: 0, y: 0 });

  const count = 2000;
  
  const [positions, colors] = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    
    // Nebula color gradients: purple, blue, teal
    const colorOptions = [
      new THREE.Color('#9b5de5'), // purple
      new THREE.Color('#00f5d4'), // teal
      new THREE.Color('#00bbf9'), // blue
      new THREE.Color('#ffffff'), // white
    ];

    for (let i = 0; i < count; i++) {
      // Create stars in a sphere
      const r = 40 + Math.random() * 60; // radius between 40 and 100
      const theta = 2 * Math.PI * Math.random();
      const phi = Math.acos(2 * Math.random() - 1);
      
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);

      const color = colorOptions[Math.floor(Math.random() * colorOptions.length)];
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
    
    return [positions, colors];
  }, [count]);

  useEffect(() => {
    let subscription: any;
    Gyroscope.isAvailableAsync().then((available) => {
      if (available) {
        Gyroscope.setUpdateInterval(16); // ~60fps
        subscription = Gyroscope.addListener((data) => {
          gyroRef.current = data;
        });
      }
    });

    return () => {
      if (subscription) subscription.remove();
    };
  }, []);

  useFrame((state, delta) => {
    if (!pointsRef.current) return;

    // Slow drifting
    pointsRef.current.rotation.y += delta * 0.05;
    pointsRef.current.rotation.x += delta * 0.02;

    // Gyroscope Parallax (accumulate rotation rate, apply soft clamp and spring to center)
    let targetX = currentRotation.current.x + gyroRef.current.x * delta;
    let targetY = currentRotation.current.y + gyroRef.current.y * delta;
    
    targetX = THREE.MathUtils.clamp(targetX, -1.5, 1.5);
    targetY = THREE.MathUtils.clamp(targetY, -1.5, 1.5);
    
    currentRotation.current.x = THREE.MathUtils.lerp(targetX, 0, 0.02);
    currentRotation.current.y = THREE.MathUtils.lerp(targetY, 0, 0.02);

    state.camera.position.x = currentRotation.current.y * 10;
    state.camera.position.y = currentRotation.current.x * 10;
    state.camera.lookAt(0, 0, 0);
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.6}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation={true}
      />
    </points>
  );
}

export default function Starfield() {
  return (
    <View style={StyleSheet.absoluteFill}>
      <Canvas camera={{ position: [0, 0, 10], fov: 75 }}>
        <color attach="background" args={[Colors.background]} />
        <ambientLight intensity={0.5} />
        <Stars />
      </Canvas>
    </View>
  );
}
