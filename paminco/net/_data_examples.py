"""This module provides the XML code for some basic example networks."""


NET_ELECTRICAL_PIECEWISE = """<network>
  <edges>
    <edge from="s" to="v">
      <cost>
        <piecewisequadratic>
          <functionpart a=".5" b="0" c="0" tau="-inf"/>
          <functionpart a="2.5" b="-12" c="18" tau="3"/>
        </piecewisequadratic>
      </cost>
    </edge>
    <edge from="v" to="t">
      <cost>
        <piecewisequadratic>
          <functionpart a=".5" b="0" c="0" tau="-inf"/>
          <functionpart a="1.5" b="-4" c="4" tau="2"/>
        </piecewisequadratic>
      </cost>
    </edge>
    <edge from="s" to="t">
      <cost>
        <piecewisequadratic>
          <functionpart a=".5" b="0" c="0" tau="-inf"/>
          <functionpart a="2" b="-3" c="1.5" tau="1"/>
        </piecewisequadratic>
      </cost>
    </edge>
  </edges>
  <commodities>
    <commodity from="s" to="t" rate="1.0" />
  </commodities>
  <nodes>
    <node node="s" x="0" y="0" />
    <node node="v" x="1" y="1" />
    <node node="t" x="2" y="0" />
  </nodes>
</network>"""


NET_ELECTRICAL_BRAESS = """<network>
  <edges>
    <edge from="s" to="v">
      <cost>
        <piecewisequadratic>
          <functionpart a="1" b="0" c="0" tau="-inf"/>
        </piecewisequadratic>
      </cost>
    </edge>
    <edge from="s" to="w">
      <cost>
        <piecewisequadratic>
          <functionpart a="2.5" b="0" c="0" tau="-inf"/>
          <functionpart a="0.5" b="4" c="0" tau="1"/>
        </piecewisequadratic>
      </cost>
    </edge>
    <edge from="v" to="w">
      <cost>
        <piecewisequadratic>
          <functionpart a="2.25" b="0" c="0" tau="-inf"/>
          <functionpart a="0.5" b="0" c="0" tau="0"/>
        </piecewisequadratic>
      </cost>
    </edge>
    <edge from="v" to="t">
      <cost>
        <piecewisequadratic>
          <functionpart a="2.5" b="0" c="0" tau="-inf"/>
          <functionpart a="0.5" b="4" c="0" tau="1"/>
        </piecewisequadratic>
      </cost>
    </edge>
    <edge from="w" to="t">
      <cost>
        <piecewisequadratic>
          <functionpart a="1" b="0" c="0" tau="-inf"/>
        </piecewisequadratic>
      </cost>
    </edge>
  </edges>
  <commodities>
    <commodity from="s" to="t" rate="1.0" />
  </commodities>
  <nodes>
    <node node="s" x="0" y="0" />
    <node node="v" x="1" y="1" />
    <node node="w" x="1" y="-1" />
    <node node="t" x="2" y="0" />
  </nodes>
</network>
"""


NET_DISCONTINUOUS_COST = """<network>
  <edges>
    <edge from="s" to="v">
      <cost>
        <piecewisequadratic>
          <functionpart a="0.5" b="0" c="0" tau="-inf"/>
          <functionpart a="inf" b="2" c="0" tau="2"/>
          <functionpart a="0.5" b="1" c="0" tau="2"/>
        </piecewisequadratic>
      </cost>
    </edge>
    <edge from="v" to="t">
      <cost>
        <piecewisequadratic>
          <functionpart a="0.5" b="0" c="0" tau="-inf"/>
          <functionpart a="inf" b="1" c="0" tau="1"/>
          <functionpart a="1" b="1" c="0" tau="1"/>
        </piecewisequadratic>
      </cost>
    </edge>
    <edge from="s" to="t">
      <cost>
        <piecewisequadratic>
          <functionpart a="0.5" b="0" c="0" tau="-inf"/>
          <functionpart a="inf" b="3" c="0" tau="3"/>
          <functionpart a="1.5" b="-4" c="0" tau="3"/>
        </piecewisequadratic>
      </cost>
    </edge>
  </edges>
  <commodities>
    <commodity from="s" to="t" rate="1.0" />
  </commodities>
  <nodes>
    <node node="s" x="0" y="0" />
    <node node="v" x="1" y="1" />
    <node node="t" x="2" y="0" />
  </nodes>
</network>"""


NET_SIMPLE_POLYNOMIAL = """<network>
  <edges>
    <edge from="s" to="t">
      <cost>
        <polynomial>
          <coefficient i="2">3.0</coefficient>
          <coefficient i="3">1.0</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="t" to="v" lb="0" ub="inf">
      <cost>
        <polynomial signed="True">
          <coefficient i="3">1</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="v" to="w" lb="-inf" ub="inf">
      <cost>
        <polynomial signed="True">
          <coefficient i="3">1</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="w" to="u" lb="-1000" ub="inf">
      <cost>
        <polynomial signed="True">
          <coefficient i="3">1</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="u" to="u1" lb="-inf" ub="1000">
      <cost>
        <polynomial signed="True">
          <coefficient i="3">1</coefficient>
        </polynomial>
      </cost>
    </edge>
    <edge from="u1" to="u2" lb="-1000" ub="1000">
      <cost>
        <polynomial signed="True">
          <coefficient i="3">1</coefficient>
        </polynomial>
      </cost>
    </edge>
  </edges>
  <commodities>
    <commodity from="s" to="t" rate="1.0" />
  </commodities>
  <nodes>
    <node node="s" x="0" y="0" />
    <node node="t" x="2" y="0" />
    <node node="v" x="4" y="0" />
    <node node="w" x="6" y="0" />
    <node node="u" x="8" y="0" />
    <node node="u1" x="10" y="0" />
    <node node="u2" x="12" y="0" />
  </nodes>
</network>
"""