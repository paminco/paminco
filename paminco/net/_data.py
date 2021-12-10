"""

Data source for SiouxFalls:
Transportation Networks for Research Core Team. Transportation Networks for Research.
https://github.com/bstabler/TransportationNetworks. Accessed 11, 30, 2021.
"""
import numpy as np


XML_SIOUX = (
"""<?xml version="1.0" ?>
<network>
  <edges>
    <edge from="1" to="2">
      <cost>
        <polynomial>
          <coefficient i="0">6.0</coefficient>
          <coefficient i="4">2.0000000003439404e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="1" to="3">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">2.0000000010963015e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="2" to="1">
      <cost>
        <polynomial>
          <coefficient i="0">6.0</coefficient>
          <coefficient i="4">2.0000000003439404e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="2" to="6">
      <cost>
        <polynomial>
          <coefficient i="0">5.0</coefficient>
          <coefficient i="4">1.2409999999794962e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="3" to="1">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">2.0000000010963015e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="3" to="4">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">6.999999995786051e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="3" to="12">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">2.0000000010963015e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="4" to="3">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">6.999999995786051e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="4" to="5">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">3.0000000002626555e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="4" to="11">
      <cost>
        <polynomial>
          <coefficient i="0">6.0</coefficient>
          <coefficient i="4">1.5500000002460378e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="5" to="4">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">3.0000000002626555e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="5" to="6">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">1.0009999996575303e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="5" to="9">
      <cost>
        <polynomial>
          <coefficient i="0">5.0</coefficient>
          <coefficient i="4">7.5e-17</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="6" to="2">
      <cost>
        <polynomial>
          <coefficient i="0">5.0</coefficient>
          <coefficient i="4">1.2409999999794962e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="6" to="5">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">1.0009999996575303e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="6" to="8">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">5.209999998180762e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="7" to="8">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">1.190000000199907e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="7" to="18">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">1.0000000005481507e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="8" to="6">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">5.209999998180762e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="8" to="7">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">1.190000000199907e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="8" to="9">
      <cost>
        <polynomial>
          <coefficient i="0">10.0</coefficient>
          <coefficient i="4">2.306000000389088e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="8" to="16">
      <cost>
        <polynomial>
          <coefficient i="0">5.0</coefficient>
          <coefficient i="4">1.156999999572462e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="9" to="5">
      <cost>
        <polynomial>
          <coefficient i="0">5.0</coefficient>
          <coefficient i="4">7.5e-17</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="9" to="8">
      <cost>
        <polynomial>
          <coefficient i="0">10.0</coefficient>
          <coefficient i="4">2.306000000389088e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="9" to="10">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">1.1999999995062995e-17</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="10" to="9">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">1.1999999995062995e-17</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="10" to="11">
      <cost>
        <polynomial>
          <coefficient i="0">5.0</coefficient>
          <coefficient i="4">7.5e-17</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="10" to="15">
      <cost>
        <polynomial>
          <coefficient i="0">6.0</coefficient>
          <coefficient i="4">2.6999999984576458e-17</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="10" to="16">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">1.0800000000651648e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="10" to="17">
      <cost>
        <polynomial>
          <coefficient i="0">8.0</coefficient>
          <coefficient i="4">1.929999999475087e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="11" to="4">
      <cost>
        <polynomial>
          <coefficient i="0">6.0</coefficient>
          <coefficient i="4">1.5500000002460378e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="11" to="10">
      <cost>
        <polynomial>
          <coefficient i="0">5.0</coefficient>
          <coefficient i="4">7.5e-17</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="11" to="12">
      <cost>
        <polynomial>
          <coefficient i="0">6.0</coefficient>
          <coefficient i="4">1.5500000002460378e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="11" to="14">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">1.0610000003948405e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="12" to="3">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">2.0000000010963015e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="12" to="11">
      <cost>
        <polynomial>
          <coefficient i="0">6.0</coefficient>
          <coefficient i="4">1.5500000002460378e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="12" to="13">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">1.0000000001719702e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="13" to="12">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">1.0000000001719702e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="13" to="24">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">8.93000000010843e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="14" to="11">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">1.0610000003948405e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="14" to="15">
      <cost>
        <polynomial>
          <coefficient i="0">5.0</coefficient>
          <coefficient i="4">1.0850000001199862e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="14" to="23">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">1.0200000000451704e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="15" to="10">
      <cost>
        <polynomial>
          <coefficient i="0">6.0</coefficient>
          <coefficient i="4">2.6999999984576458e-17</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="15" to="14">
      <cost>
        <polynomial>
          <coefficient i="0">5.0</coefficient>
          <coefficient i="4">1.0850000001199862e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="15" to="19">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">1.0000000003349737e-17</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="15" to="22">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">5.3000000006463875e-17</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="16" to="8">
      <cost>
        <polynomial>
          <coefficient i="0">5.0</coefficient>
          <coefficient i="4">1.156999999572462e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="16" to="10">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">1.0800000000651648e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="16" to="17">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">4.010000000338166e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="16" to="18">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">3.0000000016184863e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="17" to="10">
      <cost>
        <polynomial>
          <coefficient i="0">8.0</coefficient>
          <coefficient i="4">1.929999999475087e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="17" to="16">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">4.010000000338166e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="17" to="19">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">5.539999998417703e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="18" to="7">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">1.0000000005481507e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="18" to="16">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">3.0000000016184863e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="18" to="20">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">2.0000000010963015e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="19" to="15">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">1.0000000003349737e-17</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="19" to="17">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">5.539999998417703e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="19" to="20">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">9.579999998589453e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="20" to="18">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">2.0000000010963015e-18</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="20" to="19">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">9.579999998589453e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="20" to="21">
      <cost>
        <polynomial>
          <coefficient i="0">6.0</coefficient>
          <coefficient i="4">1.3730000005271283e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="20" to="22">
      <cost>
        <polynomial>
          <coefficient i="0">5.0</coefficient>
          <coefficient i="4">1.1300000001271254e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="21" to="20">
      <cost>
        <polynomial>
          <coefficient i="0">6.0</coefficient>
          <coefficient i="4">1.3730000005271283e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="21" to="22">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">4.010000000338166e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="21" to="24">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">7.899999998106579e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="22" to="15">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">5.3000000006463875e-17</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="22" to="20">
      <cost>
        <polynomial>
          <coefficient i="0">5.0</coefficient>
          <coefficient i="4">1.1300000001271254e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="22" to="21">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">4.010000000338166e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="22" to="23">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">9.6e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="23" to="14">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">1.0200000000451704e-15</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="23" to="22">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">9.6e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="23" to="24">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">4.510000000315091e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="24" to="13">
      <cost>
        <polynomial>
          <coefficient i="0">4.0</coefficient>
          <coefficient i="4">8.93000000010843e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="24" to="21">
      <cost>
        <polynomial>
          <coefficient i="0">3.0</coefficient>
          <coefficient i="4">7.899999998106579e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="24" to="23">
      <cost>
        <polynomial>
          <coefficient i="0">2.0</coefficient>
          <coefficient i="4">4.510000000315091e-16</coefficient>
        </polynomial>
      </cost>
    </edge>
  </edges>
  <commodities>
    <commodity from="1" to="1" rate="0.0"/>
    <commodity from="1" to="2" rate="100.0"/>
    <commodity from="1" to="3" rate="100.0"/>
    <commodity from="1" to="4" rate="500.0"/>
    <commodity from="1" to="5" rate="200.0"/>
    <commodity from="1" to="6" rate="300.0"/>
    <commodity from="1" to="7" rate="500.0"/>
    <commodity from="1" to="8" rate="800.0"/>
    <commodity from="1" to="9" rate="500.0"/>
    <commodity from="1" to="10" rate="1300.0"/>
    <commodity from="1" to="11" rate="500.0"/>
    <commodity from="1" to="12" rate="200.0"/>
    <commodity from="1" to="13" rate="500.0"/>
    <commodity from="1" to="14" rate="300.0"/>
    <commodity from="1" to="15" rate="500.0"/>
    <commodity from="1" to="16" rate="500.0"/>
    <commodity from="1" to="17" rate="400.0"/>
    <commodity from="1" to="18" rate="100.0"/>
    <commodity from="1" to="19" rate="300.0"/>
    <commodity from="1" to="20" rate="300.0"/>
    <commodity from="1" to="21" rate="100.0"/>
    <commodity from="1" to="22" rate="400.0"/>
    <commodity from="1" to="23" rate="300.0"/>
    <commodity from="1" to="24" rate="100.0"/>
    <commodity from="2" to="1" rate="100.0"/>
    <commodity from="2" to="2" rate="0.0"/>
    <commodity from="2" to="3" rate="100.0"/>
    <commodity from="2" to="4" rate="200.0"/>
    <commodity from="2" to="5" rate="100.0"/>
    <commodity from="2" to="6" rate="400.0"/>
    <commodity from="2" to="7" rate="200.0"/>
    <commodity from="2" to="8" rate="400.0"/>
    <commodity from="2" to="9" rate="200.0"/>
    <commodity from="2" to="10" rate="600.0"/>
    <commodity from="2" to="11" rate="200.0"/>
    <commodity from="2" to="12" rate="100.0"/>
    <commodity from="2" to="13" rate="300.0"/>
    <commodity from="2" to="14" rate="100.0"/>
    <commodity from="2" to="15" rate="100.0"/>
    <commodity from="2" to="16" rate="400.0"/>
    <commodity from="2" to="17" rate="200.0"/>
    <commodity from="2" to="18" rate="0.0"/>
    <commodity from="2" to="19" rate="100.0"/>
    <commodity from="2" to="20" rate="100.0"/>
    <commodity from="2" to="21" rate="0.0"/>
    <commodity from="2" to="22" rate="100.0"/>
    <commodity from="2" to="23" rate="0.0"/>
    <commodity from="2" to="24" rate="0.0"/>
    <commodity from="3" to="1" rate="100.0"/>
    <commodity from="3" to="2" rate="100.0"/>
    <commodity from="3" to="3" rate="0.0"/>
    <commodity from="3" to="4" rate="200.0"/>
    <commodity from="3" to="5" rate="100.0"/>
    <commodity from="3" to="6" rate="300.0"/>
    <commodity from="3" to="7" rate="100.0"/>
    <commodity from="3" to="8" rate="200.0"/>
    <commodity from="3" to="9" rate="100.0"/>
    <commodity from="3" to="10" rate="300.0"/>
    <commodity from="3" to="11" rate="300.0"/>
    <commodity from="3" to="12" rate="200.0"/>
    <commodity from="3" to="13" rate="100.0"/>
    <commodity from="3" to="14" rate="100.0"/>
    <commodity from="3" to="15" rate="100.0"/>
    <commodity from="3" to="16" rate="200.0"/>
    <commodity from="3" to="17" rate="100.0"/>
    <commodity from="3" to="18" rate="0.0"/>
    <commodity from="3" to="19" rate="0.0"/>
    <commodity from="3" to="20" rate="0.0"/>
    <commodity from="3" to="21" rate="0.0"/>
    <commodity from="3" to="22" rate="100.0"/>
    <commodity from="3" to="23" rate="100.0"/>
    <commodity from="3" to="24" rate="0.0"/>
    <commodity from="4" to="1" rate="500.0"/>
    <commodity from="4" to="2" rate="200.0"/>
    <commodity from="4" to="3" rate="200.0"/>
    <commodity from="4" to="4" rate="0.0"/>
    <commodity from="4" to="5" rate="500.0"/>
    <commodity from="4" to="6" rate="400.0"/>
    <commodity from="4" to="7" rate="400.0"/>
    <commodity from="4" to="8" rate="700.0"/>
    <commodity from="4" to="9" rate="700.0"/>
    <commodity from="4" to="10" rate="1200.0"/>
    <commodity from="4" to="11" rate="1400.0"/>
    <commodity from="4" to="12" rate="600.0"/>
    <commodity from="4" to="13" rate="600.0"/>
    <commodity from="4" to="14" rate="500.0"/>
    <commodity from="4" to="15" rate="500.0"/>
    <commodity from="4" to="16" rate="800.0"/>
    <commodity from="4" to="17" rate="500.0"/>
    <commodity from="4" to="18" rate="100.0"/>
    <commodity from="4" to="19" rate="200.0"/>
    <commodity from="4" to="20" rate="300.0"/>
    <commodity from="4" to="21" rate="200.0"/>
    <commodity from="4" to="22" rate="400.0"/>
    <commodity from="4" to="23" rate="500.0"/>
    <commodity from="4" to="24" rate="200.0"/>
    <commodity from="5" to="1" rate="200.0"/>
    <commodity from="5" to="2" rate="100.0"/>
    <commodity from="5" to="3" rate="100.0"/>
    <commodity from="5" to="4" rate="500.0"/>
    <commodity from="5" to="5" rate="0.0"/>
    <commodity from="5" to="6" rate="200.0"/>
    <commodity from="5" to="7" rate="200.0"/>
    <commodity from="5" to="8" rate="500.0"/>
    <commodity from="5" to="9" rate="800.0"/>
    <commodity from="5" to="10" rate="1000.0"/>
    <commodity from="5" to="11" rate="500.0"/>
    <commodity from="5" to="12" rate="200.0"/>
    <commodity from="5" to="13" rate="200.0"/>
    <commodity from="5" to="14" rate="100.0"/>
    <commodity from="5" to="15" rate="200.0"/>
    <commodity from="5" to="16" rate="500.0"/>
    <commodity from="5" to="17" rate="200.0"/>
    <commodity from="5" to="18" rate="0.0"/>
    <commodity from="5" to="19" rate="100.0"/>
    <commodity from="5" to="20" rate="100.0"/>
    <commodity from="5" to="21" rate="100.0"/>
    <commodity from="5" to="22" rate="200.0"/>
    <commodity from="5" to="23" rate="100.0"/>
    <commodity from="5" to="24" rate="0.0"/>
    <commodity from="6" to="1" rate="300.0"/>
    <commodity from="6" to="2" rate="400.0"/>
    <commodity from="6" to="3" rate="300.0"/>
    <commodity from="6" to="4" rate="400.0"/>
    <commodity from="6" to="5" rate="200.0"/>
    <commodity from="6" to="6" rate="0.0"/>
    <commodity from="6" to="7" rate="400.0"/>
    <commodity from="6" to="8" rate="800.0"/>
    <commodity from="6" to="9" rate="400.0"/>
    <commodity from="6" to="10" rate="800.0"/>
    <commodity from="6" to="11" rate="400.0"/>
    <commodity from="6" to="12" rate="200.0"/>
    <commodity from="6" to="13" rate="200.0"/>
    <commodity from="6" to="14" rate="100.0"/>
    <commodity from="6" to="15" rate="200.0"/>
    <commodity from="6" to="16" rate="900.0"/>
    <commodity from="6" to="17" rate="500.0"/>
    <commodity from="6" to="18" rate="100.0"/>
    <commodity from="6" to="19" rate="200.0"/>
    <commodity from="6" to="20" rate="300.0"/>
    <commodity from="6" to="21" rate="100.0"/>
    <commodity from="6" to="22" rate="200.0"/>
    <commodity from="6" to="23" rate="100.0"/>
    <commodity from="6" to="24" rate="100.0"/>
    <commodity from="7" to="1" rate="500.0"/>
    <commodity from="7" to="2" rate="200.0"/>
    <commodity from="7" to="3" rate="100.0"/>
    <commodity from="7" to="4" rate="400.0"/>
    <commodity from="7" to="5" rate="200.0"/>
    <commodity from="7" to="6" rate="400.0"/>
    <commodity from="7" to="7" rate="0.0"/>
    <commodity from="7" to="8" rate="1000.0"/>
    <commodity from="7" to="9" rate="600.0"/>
    <commodity from="7" to="10" rate="1900.0"/>
    <commodity from="7" to="11" rate="500.0"/>
    <commodity from="7" to="12" rate="700.0"/>
    <commodity from="7" to="13" rate="400.0"/>
    <commodity from="7" to="14" rate="200.0"/>
    <commodity from="7" to="15" rate="500.0"/>
    <commodity from="7" to="16" rate="1400.0"/>
    <commodity from="7" to="17" rate="1000.0"/>
    <commodity from="7" to="18" rate="200.0"/>
    <commodity from="7" to="19" rate="400.0"/>
    <commodity from="7" to="20" rate="500.0"/>
    <commodity from="7" to="21" rate="200.0"/>
    <commodity from="7" to="22" rate="500.0"/>
    <commodity from="7" to="23" rate="200.0"/>
    <commodity from="7" to="24" rate="100.0"/>
    <commodity from="8" to="1" rate="800.0"/>
    <commodity from="8" to="2" rate="400.0"/>
    <commodity from="8" to="3" rate="200.0"/>
    <commodity from="8" to="4" rate="700.0"/>
    <commodity from="8" to="5" rate="500.0"/>
    <commodity from="8" to="6" rate="800.0"/>
    <commodity from="8" to="7" rate="1000.0"/>
    <commodity from="8" to="8" rate="0.0"/>
    <commodity from="8" to="9" rate="800.0"/>
    <commodity from="8" to="10" rate="1600.0"/>
    <commodity from="8" to="11" rate="800.0"/>
    <commodity from="8" to="12" rate="600.0"/>
    <commodity from="8" to="13" rate="600.0"/>
    <commodity from="8" to="14" rate="400.0"/>
    <commodity from="8" to="15" rate="600.0"/>
    <commodity from="8" to="16" rate="2200.0"/>
    <commodity from="8" to="17" rate="1400.0"/>
    <commodity from="8" to="18" rate="300.0"/>
    <commodity from="8" to="19" rate="700.0"/>
    <commodity from="8" to="20" rate="900.0"/>
    <commodity from="8" to="21" rate="400.0"/>
    <commodity from="8" to="22" rate="500.0"/>
    <commodity from="8" to="23" rate="300.0"/>
    <commodity from="8" to="24" rate="200.0"/>
    <commodity from="9" to="1" rate="500.0"/>
    <commodity from="9" to="2" rate="200.0"/>
    <commodity from="9" to="3" rate="100.0"/>
    <commodity from="9" to="4" rate="700.0"/>
    <commodity from="9" to="5" rate="800.0"/>
    <commodity from="9" to="6" rate="400.0"/>
    <commodity from="9" to="7" rate="600.0"/>
    <commodity from="9" to="8" rate="800.0"/>
    <commodity from="9" to="9" rate="0.0"/>
    <commodity from="9" to="10" rate="2800.0"/>
    <commodity from="9" to="11" rate="1400.0"/>
    <commodity from="9" to="12" rate="600.0"/>
    <commodity from="9" to="13" rate="600.0"/>
    <commodity from="9" to="14" rate="600.0"/>
    <commodity from="9" to="15" rate="900.0"/>
    <commodity from="9" to="16" rate="1400.0"/>
    <commodity from="9" to="17" rate="900.0"/>
    <commodity from="9" to="18" rate="200.0"/>
    <commodity from="9" to="19" rate="400.0"/>
    <commodity from="9" to="20" rate="600.0"/>
    <commodity from="9" to="21" rate="300.0"/>
    <commodity from="9" to="22" rate="700.0"/>
    <commodity from="9" to="23" rate="500.0"/>
    <commodity from="9" to="24" rate="200.0"/>
    <commodity from="10" to="1" rate="1300.0"/>
    <commodity from="10" to="2" rate="600.0"/>
    <commodity from="10" to="3" rate="300.0"/>
    <commodity from="10" to="4" rate="1200.0"/>
    <commodity from="10" to="5" rate="1000.0"/>
    <commodity from="10" to="6" rate="800.0"/>
    <commodity from="10" to="7" rate="1900.0"/>
    <commodity from="10" to="8" rate="1600.0"/>
    <commodity from="10" to="9" rate="2800.0"/>
    <commodity from="10" to="10" rate="0.0"/>
    <commodity from="10" to="11" rate="4000.0"/>
    <commodity from="10" to="12" rate="2000.0"/>
    <commodity from="10" to="13" rate="1900.0"/>
    <commodity from="10" to="14" rate="2100.0"/>
    <commodity from="10" to="15" rate="4000.0"/>
    <commodity from="10" to="16" rate="4400.0"/>
    <commodity from="10" to="17" rate="3900.0"/>
    <commodity from="10" to="18" rate="700.0"/>
    <commodity from="10" to="19" rate="1800.0"/>
    <commodity from="10" to="20" rate="2500.0"/>
    <commodity from="10" to="21" rate="1200.0"/>
    <commodity from="10" to="22" rate="2600.0"/>
    <commodity from="10" to="23" rate="1800.0"/>
    <commodity from="10" to="24" rate="800.0"/>
    <commodity from="11" to="1" rate="500.0"/>
    <commodity from="11" to="2" rate="200.0"/>
    <commodity from="11" to="3" rate="300.0"/>
    <commodity from="11" to="4" rate="1500.0"/>
    <commodity from="11" to="5" rate="500.0"/>
    <commodity from="11" to="6" rate="400.0"/>
    <commodity from="11" to="7" rate="500.0"/>
    <commodity from="11" to="8" rate="800.0"/>
    <commodity from="11" to="9" rate="1400.0"/>
    <commodity from="11" to="10" rate="3900.0"/>
    <commodity from="11" to="11" rate="0.0"/>
    <commodity from="11" to="12" rate="1400.0"/>
    <commodity from="11" to="13" rate="1000.0"/>
    <commodity from="11" to="14" rate="1600.0"/>
    <commodity from="11" to="15" rate="1400.0"/>
    <commodity from="11" to="16" rate="1400.0"/>
    <commodity from="11" to="17" rate="1000.0"/>
    <commodity from="11" to="18" rate="100.0"/>
    <commodity from="11" to="19" rate="400.0"/>
    <commodity from="11" to="20" rate="600.0"/>
    <commodity from="11" to="21" rate="400.0"/>
    <commodity from="11" to="22" rate="1100.0"/>
    <commodity from="11" to="23" rate="1300.0"/>
    <commodity from="11" to="24" rate="600.0"/>
    <commodity from="12" to="1" rate="200.0"/>
    <commodity from="12" to="2" rate="100.0"/>
    <commodity from="12" to="3" rate="200.0"/>
    <commodity from="12" to="4" rate="600.0"/>
    <commodity from="12" to="5" rate="200.0"/>
    <commodity from="12" to="6" rate="200.0"/>
    <commodity from="12" to="7" rate="700.0"/>
    <commodity from="12" to="8" rate="600.0"/>
    <commodity from="12" to="9" rate="600.0"/>
    <commodity from="12" to="10" rate="2000.0"/>
    <commodity from="12" to="11" rate="1400.0"/>
    <commodity from="12" to="12" rate="0.0"/>
    <commodity from="12" to="13" rate="1300.0"/>
    <commodity from="12" to="14" rate="700.0"/>
    <commodity from="12" to="15" rate="700.0"/>
    <commodity from="12" to="16" rate="700.0"/>
    <commodity from="12" to="17" rate="600.0"/>
    <commodity from="12" to="18" rate="200.0"/>
    <commodity from="12" to="19" rate="300.0"/>
    <commodity from="12" to="20" rate="400.0"/>
    <commodity from="12" to="21" rate="300.0"/>
    <commodity from="12" to="22" rate="700.0"/>
    <commodity from="12" to="23" rate="700.0"/>
    <commodity from="12" to="24" rate="500.0"/>
    <commodity from="13" to="1" rate="500.0"/>
    <commodity from="13" to="2" rate="300.0"/>
    <commodity from="13" to="3" rate="100.0"/>
    <commodity from="13" to="4" rate="600.0"/>
    <commodity from="13" to="5" rate="200.0"/>
    <commodity from="13" to="6" rate="200.0"/>
    <commodity from="13" to="7" rate="400.0"/>
    <commodity from="13" to="8" rate="600.0"/>
    <commodity from="13" to="9" rate="600.0"/>
    <commodity from="13" to="10" rate="1900.0"/>
    <commodity from="13" to="11" rate="1000.0"/>
    <commodity from="13" to="12" rate="1300.0"/>
    <commodity from="13" to="13" rate="0.0"/>
    <commodity from="13" to="14" rate="600.0"/>
    <commodity from="13" to="15" rate="700.0"/>
    <commodity from="13" to="16" rate="600.0"/>
    <commodity from="13" to="17" rate="500.0"/>
    <commodity from="13" to="18" rate="100.0"/>
    <commodity from="13" to="19" rate="300.0"/>
    <commodity from="13" to="20" rate="600.0"/>
    <commodity from="13" to="21" rate="600.0"/>
    <commodity from="13" to="22" rate="1300.0"/>
    <commodity from="13" to="23" rate="800.0"/>
    <commodity from="13" to="24" rate="800.0"/>
    <commodity from="14" to="1" rate="300.0"/>
    <commodity from="14" to="2" rate="100.0"/>
    <commodity from="14" to="3" rate="100.0"/>
    <commodity from="14" to="4" rate="500.0"/>
    <commodity from="14" to="5" rate="100.0"/>
    <commodity from="14" to="6" rate="100.0"/>
    <commodity from="14" to="7" rate="200.0"/>
    <commodity from="14" to="8" rate="400.0"/>
    <commodity from="14" to="9" rate="600.0"/>
    <commodity from="14" to="10" rate="2100.0"/>
    <commodity from="14" to="11" rate="1600.0"/>
    <commodity from="14" to="12" rate="700.0"/>
    <commodity from="14" to="13" rate="600.0"/>
    <commodity from="14" to="14" rate="0.0"/>
    <commodity from="14" to="15" rate="1300.0"/>
    <commodity from="14" to="16" rate="700.0"/>
    <commodity from="14" to="17" rate="700.0"/>
    <commodity from="14" to="18" rate="100.0"/>
    <commodity from="14" to="19" rate="300.0"/>
    <commodity from="14" to="20" rate="500.0"/>
    <commodity from="14" to="21" rate="400.0"/>
    <commodity from="14" to="22" rate="1200.0"/>
    <commodity from="14" to="23" rate="1100.0"/>
    <commodity from="14" to="24" rate="400.0"/>
    <commodity from="15" to="1" rate="500.0"/>
    <commodity from="15" to="2" rate="100.0"/>
    <commodity from="15" to="3" rate="100.0"/>
    <commodity from="15" to="4" rate="500.0"/>
    <commodity from="15" to="5" rate="200.0"/>
    <commodity from="15" to="6" rate="200.0"/>
    <commodity from="15" to="7" rate="500.0"/>
    <commodity from="15" to="8" rate="600.0"/>
    <commodity from="15" to="9" rate="1000.0"/>
    <commodity from="15" to="10" rate="4000.0"/>
    <commodity from="15" to="11" rate="1400.0"/>
    <commodity from="15" to="12" rate="700.0"/>
    <commodity from="15" to="13" rate="700.0"/>
    <commodity from="15" to="14" rate="1300.0"/>
    <commodity from="15" to="15" rate="0.0"/>
    <commodity from="15" to="16" rate="1200.0"/>
    <commodity from="15" to="17" rate="1500.0"/>
    <commodity from="15" to="18" rate="200.0"/>
    <commodity from="15" to="19" rate="800.0"/>
    <commodity from="15" to="20" rate="1100.0"/>
    <commodity from="15" to="21" rate="800.0"/>
    <commodity from="15" to="22" rate="2600.0"/>
    <commodity from="15" to="23" rate="1000.0"/>
    <commodity from="15" to="24" rate="400.0"/>
    <commodity from="16" to="1" rate="500.0"/>
    <commodity from="16" to="2" rate="400.0"/>
    <commodity from="16" to="3" rate="200.0"/>
    <commodity from="16" to="4" rate="800.0"/>
    <commodity from="16" to="5" rate="500.0"/>
    <commodity from="16" to="6" rate="900.0"/>
    <commodity from="16" to="7" rate="1400.0"/>
    <commodity from="16" to="8" rate="2200.0"/>
    <commodity from="16" to="9" rate="1400.0"/>
    <commodity from="16" to="10" rate="4400.0"/>
    <commodity from="16" to="11" rate="1400.0"/>
    <commodity from="16" to="12" rate="700.0"/>
    <commodity from="16" to="13" rate="600.0"/>
    <commodity from="16" to="14" rate="700.0"/>
    <commodity from="16" to="15" rate="1200.0"/>
    <commodity from="16" to="16" rate="0.0"/>
    <commodity from="16" to="17" rate="2800.0"/>
    <commodity from="16" to="18" rate="500.0"/>
    <commodity from="16" to="19" rate="1300.0"/>
    <commodity from="16" to="20" rate="1600.0"/>
    <commodity from="16" to="21" rate="600.0"/>
    <commodity from="16" to="22" rate="1200.0"/>
    <commodity from="16" to="23" rate="500.0"/>
    <commodity from="16" to="24" rate="300.0"/>
    <commodity from="17" to="1" rate="400.0"/>
    <commodity from="17" to="2" rate="200.0"/>
    <commodity from="17" to="3" rate="100.0"/>
    <commodity from="17" to="4" rate="500.0"/>
    <commodity from="17" to="5" rate="200.0"/>
    <commodity from="17" to="6" rate="500.0"/>
    <commodity from="17" to="7" rate="1000.0"/>
    <commodity from="17" to="8" rate="1400.0"/>
    <commodity from="17" to="9" rate="900.0"/>
    <commodity from="17" to="10" rate="3900.0"/>
    <commodity from="17" to="11" rate="1000.0"/>
    <commodity from="17" to="12" rate="600.0"/>
    <commodity from="17" to="13" rate="500.0"/>
    <commodity from="17" to="14" rate="700.0"/>
    <commodity from="17" to="15" rate="1500.0"/>
    <commodity from="17" to="16" rate="2800.0"/>
    <commodity from="17" to="17" rate="0.0"/>
    <commodity from="17" to="18" rate="600.0"/>
    <commodity from="17" to="19" rate="1700.0"/>
    <commodity from="17" to="20" rate="1700.0"/>
    <commodity from="17" to="21" rate="600.0"/>
    <commodity from="17" to="22" rate="1700.0"/>
    <commodity from="17" to="23" rate="600.0"/>
    <commodity from="17" to="24" rate="300.0"/>
    <commodity from="18" to="1" rate="100.0"/>
    <commodity from="18" to="2" rate="0.0"/>
    <commodity from="18" to="3" rate="0.0"/>
    <commodity from="18" to="4" rate="100.0"/>
    <commodity from="18" to="5" rate="0.0"/>
    <commodity from="18" to="6" rate="100.0"/>
    <commodity from="18" to="7" rate="200.0"/>
    <commodity from="18" to="8" rate="300.0"/>
    <commodity from="18" to="9" rate="200.0"/>
    <commodity from="18" to="10" rate="700.0"/>
    <commodity from="18" to="11" rate="200.0"/>
    <commodity from="18" to="12" rate="200.0"/>
    <commodity from="18" to="13" rate="100.0"/>
    <commodity from="18" to="14" rate="100.0"/>
    <commodity from="18" to="15" rate="200.0"/>
    <commodity from="18" to="16" rate="500.0"/>
    <commodity from="18" to="17" rate="600.0"/>
    <commodity from="18" to="18" rate="0.0"/>
    <commodity from="18" to="19" rate="300.0"/>
    <commodity from="18" to="20" rate="400.0"/>
    <commodity from="18" to="21" rate="100.0"/>
    <commodity from="18" to="22" rate="300.0"/>
    <commodity from="18" to="23" rate="100.0"/>
    <commodity from="18" to="24" rate="0.0"/>
    <commodity from="19" to="1" rate="300.0"/>
    <commodity from="19" to="2" rate="100.0"/>
    <commodity from="19" to="3" rate="0.0"/>
    <commodity from="19" to="4" rate="200.0"/>
    <commodity from="19" to="5" rate="100.0"/>
    <commodity from="19" to="6" rate="200.0"/>
    <commodity from="19" to="7" rate="400.0"/>
    <commodity from="19" to="8" rate="700.0"/>
    <commodity from="19" to="9" rate="400.0"/>
    <commodity from="19" to="10" rate="1800.0"/>
    <commodity from="19" to="11" rate="400.0"/>
    <commodity from="19" to="12" rate="300.0"/>
    <commodity from="19" to="13" rate="300.0"/>
    <commodity from="19" to="14" rate="300.0"/>
    <commodity from="19" to="15" rate="800.0"/>
    <commodity from="19" to="16" rate="1300.0"/>
    <commodity from="19" to="17" rate="1700.0"/>
    <commodity from="19" to="18" rate="300.0"/>
    <commodity from="19" to="19" rate="0.0"/>
    <commodity from="19" to="20" rate="1200.0"/>
    <commodity from="19" to="21" rate="400.0"/>
    <commodity from="19" to="22" rate="1200.0"/>
    <commodity from="19" to="23" rate="300.0"/>
    <commodity from="19" to="24" rate="100.0"/>
    <commodity from="20" to="1" rate="300.0"/>
    <commodity from="20" to="2" rate="100.0"/>
    <commodity from="20" to="3" rate="0.0"/>
    <commodity from="20" to="4" rate="300.0"/>
    <commodity from="20" to="5" rate="100.0"/>
    <commodity from="20" to="6" rate="300.0"/>
    <commodity from="20" to="7" rate="500.0"/>
    <commodity from="20" to="8" rate="900.0"/>
    <commodity from="20" to="9" rate="600.0"/>
    <commodity from="20" to="10" rate="2500.0"/>
    <commodity from="20" to="11" rate="600.0"/>
    <commodity from="20" to="12" rate="500.0"/>
    <commodity from="20" to="13" rate="600.0"/>
    <commodity from="20" to="14" rate="500.0"/>
    <commodity from="20" to="15" rate="1100.0"/>
    <commodity from="20" to="16" rate="1600.0"/>
    <commodity from="20" to="17" rate="1700.0"/>
    <commodity from="20" to="18" rate="400.0"/>
    <commodity from="20" to="19" rate="1200.0"/>
    <commodity from="20" to="20" rate="0.0"/>
    <commodity from="20" to="21" rate="1200.0"/>
    <commodity from="20" to="22" rate="2400.0"/>
    <commodity from="20" to="23" rate="700.0"/>
    <commodity from="20" to="24" rate="400.0"/>
    <commodity from="21" to="1" rate="100.0"/>
    <commodity from="21" to="2" rate="0.0"/>
    <commodity from="21" to="3" rate="0.0"/>
    <commodity from="21" to="4" rate="200.0"/>
    <commodity from="21" to="5" rate="100.0"/>
    <commodity from="21" to="6" rate="100.0"/>
    <commodity from="21" to="7" rate="200.0"/>
    <commodity from="21" to="8" rate="400.0"/>
    <commodity from="21" to="9" rate="300.0"/>
    <commodity from="21" to="10" rate="1200.0"/>
    <commodity from="21" to="11" rate="400.0"/>
    <commodity from="21" to="12" rate="300.0"/>
    <commodity from="21" to="13" rate="600.0"/>
    <commodity from="21" to="14" rate="400.0"/>
    <commodity from="21" to="15" rate="800.0"/>
    <commodity from="21" to="16" rate="600.0"/>
    <commodity from="21" to="17" rate="600.0"/>
    <commodity from="21" to="18" rate="100.0"/>
    <commodity from="21" to="19" rate="400.0"/>
    <commodity from="21" to="20" rate="1200.0"/>
    <commodity from="21" to="21" rate="0.0"/>
    <commodity from="21" to="22" rate="1800.0"/>
    <commodity from="21" to="23" rate="700.0"/>
    <commodity from="21" to="24" rate="500.0"/>
    <commodity from="22" to="1" rate="400.0"/>
    <commodity from="22" to="2" rate="100.0"/>
    <commodity from="22" to="3" rate="100.0"/>
    <commodity from="22" to="4" rate="400.0"/>
    <commodity from="22" to="5" rate="200.0"/>
    <commodity from="22" to="6" rate="200.0"/>
    <commodity from="22" to="7" rate="500.0"/>
    <commodity from="22" to="8" rate="500.0"/>
    <commodity from="22" to="9" rate="700.0"/>
    <commodity from="22" to="10" rate="2600.0"/>
    <commodity from="22" to="11" rate="1100.0"/>
    <commodity from="22" to="12" rate="700.0"/>
    <commodity from="22" to="13" rate="1300.0"/>
    <commodity from="22" to="14" rate="1200.0"/>
    <commodity from="22" to="15" rate="2600.0"/>
    <commodity from="22" to="16" rate="1200.0"/>
    <commodity from="22" to="17" rate="1700.0"/>
    <commodity from="22" to="18" rate="300.0"/>
    <commodity from="22" to="19" rate="1200.0"/>
    <commodity from="22" to="20" rate="2400.0"/>
    <commodity from="22" to="21" rate="1800.0"/>
    <commodity from="22" to="22" rate="0.0"/>
    <commodity from="22" to="23" rate="2100.0"/>
    <commodity from="22" to="24" rate="1100.0"/>
    <commodity from="23" to="1" rate="300.0"/>
    <commodity from="23" to="2" rate="0.0"/>
    <commodity from="23" to="3" rate="100.0"/>
    <commodity from="23" to="4" rate="500.0"/>
    <commodity from="23" to="5" rate="100.0"/>
    <commodity from="23" to="6" rate="100.0"/>
    <commodity from="23" to="7" rate="200.0"/>
    <commodity from="23" to="8" rate="300.0"/>
    <commodity from="23" to="9" rate="500.0"/>
    <commodity from="23" to="10" rate="1800.0"/>
    <commodity from="23" to="11" rate="1300.0"/>
    <commodity from="23" to="12" rate="700.0"/>
    <commodity from="23" to="13" rate="800.0"/>
    <commodity from="23" to="14" rate="1100.0"/>
    <commodity from="23" to="15" rate="1000.0"/>
    <commodity from="23" to="16" rate="500.0"/>
    <commodity from="23" to="17" rate="600.0"/>
    <commodity from="23" to="18" rate="100.0"/>
    <commodity from="23" to="19" rate="300.0"/>
    <commodity from="23" to="20" rate="700.0"/>
    <commodity from="23" to="21" rate="700.0"/>
    <commodity from="23" to="22" rate="2100.0"/>
    <commodity from="23" to="23" rate="0.0"/>
    <commodity from="23" to="24" rate="700.0"/>
    <commodity from="24" to="1" rate="100.0"/>
    <commodity from="24" to="2" rate="0.0"/>
    <commodity from="24" to="3" rate="0.0"/>
    <commodity from="24" to="4" rate="200.0"/>
    <commodity from="24" to="5" rate="0.0"/>
    <commodity from="24" to="6" rate="100.0"/>
    <commodity from="24" to="7" rate="100.0"/>
    <commodity from="24" to="8" rate="200.0"/>
    <commodity from="24" to="9" rate="200.0"/>
    <commodity from="24" to="10" rate="800.0"/>
    <commodity from="24" to="11" rate="600.0"/>
    <commodity from="24" to="12" rate="500.0"/>
    <commodity from="24" to="13" rate="700.0"/>
    <commodity from="24" to="14" rate="400.0"/>
    <commodity from="24" to="15" rate="400.0"/>
    <commodity from="24" to="16" rate="300.0"/>
    <commodity from="24" to="17" rate="300.0"/>
    <commodity from="24" to="18" rate="0.0"/>
    <commodity from="24" to="19" rate="100.0"/>
    <commodity from="24" to="20" rate="400.0"/>
    <commodity from="24" to="21" rate="500.0"/>
    <commodity from="24" to="22" rate="1100.0"/>
    <commodity from="24" to="23" rate="700.0"/>
    <commodity from="24" to="24" rate="0.0"/>
  </commodities>
  <nodes>
    <node node="1" x="-96.77041974" y="43.61282792"/>
    <node node="2" x="-96.71125063" y="43.60581298"/>
    <node node="3" x="-96.77430341" y="43.5729616"/>
    <node node="4" x="-96.74716843" y="43.56365362"/>
    <node node="5" x="-96.73156909" y="43.56403357"/>
    <node node="6" x="-96.71164389" y="43.58758553"/>
    <node node="7" x="-96.69342281" y="43.5638436"/>
    <node node="8" x="-96.71138171" y="43.56232379"/>
    <node node="9" x="-96.73124137" y="43.54859634"/>
    <node node="10" x="-96.73143801" y="43.54527088"/>
    <node node="11" x="-96.74684071" y="43.54413068"/>
    <node node="12" x="-96.78013678" y="43.54394065"/>
    <node node="13" x="-96.79337655" y="43.49070718"/>
    <node node="14" x="-96.75103549" y="43.52930613"/>
    <node node="15" x="-96.73150355" y="43.52940117"/>
    <node node="16" x="-96.71138171" y="43.54674361"/>
    <node node="17" x="-96.71138171" y="43.54128009"/>
    <node node="18" x="-96.69407825" y="43.54674361"/>
    <node node="19" x="-96.71131617" y="43.52959125"/>
    <node node="20" x="-96.71118508" y="43.5153335"/>
    <node node="21" x="-96.7309792" y="43.51048509"/>
    <node node="22" x="-96.73124137" y="43.51485818"/>
    <node node="23" x="-96.75090441" y="43.51485818"/>
    <node node="24" x="-96.74920028" y="43.50316422"/>
  </nodes>
  <metadata NUMBER_OF_ZONES="24" NUMBER_OF_NODES="24" FIRST_THRU_NODE="1" NUMBER_OF_LINKS="76" ftn="1" TOTAL_OD_FLOW="360600"/>
</network>
""")


DATA_BRAESS = {
  "edge_data": np.array([["s", "v2"],
                         ["s", "v1"],
                         ["v2", "t"],
                         ["v1", "t"],
                         ["v1", "v2"]]),
  "cost_data": np.array([[0, 1, 0],
                         [0, 0, 0.5],
                         [0, 0, 0.5],
                         [0, 1, 0],
                         [0, 0, 0]]),
  "demand_data": ("s", "t", 10),
}
